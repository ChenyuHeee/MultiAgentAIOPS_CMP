import json
import re
from datetime import datetime, timedelta


_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")


def _parse_time_minute(value: str) -> datetime:
    s = str(value or "").strip()
    if not s:
        raise ValueError("time_minute is empty")
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    s2 = s.replace("T", " ").replace("Z", "").strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s2, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported time_minute format: {value!r}")


def _format_time_minute(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _norm_endpoint(s: str) -> str:
    tokens = re.split(r"[^a-zA-Z0-9_]+", str(s or "").strip().lower())
    drop = {"service", "svc", "api"}
    tokens = [t for t in tokens if t and t not in drop]
    return " ".join(tokens)


def _suffix_key(s: str) -> str:
    ss = str(s or "").strip()
    if "-" in ss:
        return ss.split("-", 1)[1]
    return ss

class TraceExplorer:
    def __init__(self):
        files = 'data/topology/endpoint_maps.json'
        self.endpoint_maps = self.load_data(files)

        # Indexes for endpoint naming variants.
        self._keys = list(self.endpoint_maps.keys())
        self._norm_to_keys: dict[str, list[str]] = {}
        self._suffix_to_keys: dict[str, list[str]] = {}
        for k in self._keys:
            self._norm_to_keys.setdefault(_norm_endpoint(k), []).append(k)
            self._suffix_to_keys.setdefault(_suffix_key(k), []).append(k)

    def load_data(self, filename):
        with open(filename, 'r') as f:
            return json.load(f)

    def _resolve_endpoint_keys(self, endpoint: str) -> list[str]:
        ep = str(endpoint or "").strip()
        if not ep:
            return []
        if ep in self.endpoint_maps:
            return [ep]

        keys = self._norm_to_keys.get(_norm_endpoint(ep), [])
        if keys:
            return sorted(set(keys))
        keys = self._suffix_to_keys.get(_suffix_key(ep), [])
        if keys:
            return sorted(set(keys))
        if "-" in ep:
            tail = ep.split("-", 1)[1]
            keys = self._suffix_to_keys.get(tail, [])
            if keys:
                return sorted(set(keys))
        return []

    def get_endpoint_downstream(self, endpoint, time_minute=None):
        keys = self._resolve_endpoint_keys(endpoint)
        if not keys:
            return [] if time_minute else []
        if len(keys) == 1:
            t = self.endpoint_maps.get(keys[0], {})
        else:
            # Merge multiple candidates by union.
            t = {}
            for k in keys:
                mm = self.endpoint_maps.get(k, {})
                if isinstance(mm, dict):
                    for minute, children in mm.items():
                        t.setdefault(minute, [])
                        if isinstance(children, list):
                            t[minute].extend(children)
        if time_minute:
            return t.get(time_minute, [])
        # If minute not provided, return union across all minutes.
        out = set()
        for v in t.values():
            if isinstance(v, list):
                out.update(v)
        return sorted(out)

    def get_endpoint_upstream(self, endpoint, time_minute=None):
        out = set()
        keys = set(self._resolve_endpoint_keys(endpoint) or [endpoint])
        if time_minute:
            for parent, minute_map in self.endpoint_maps.items():
                children = minute_map.get(time_minute, []) if isinstance(minute_map, dict) else []
                if isinstance(children, list) and any(k in children for k in keys):
                    out.add(parent)
            return sorted(out)
        # Union across all minutes.
        for parent, minute_map in self.endpoint_maps.items():
            if not isinstance(minute_map, dict):
                continue
            for children in minute_map.values():
                if isinstance(children, list) and any(k in children for k in keys):
                    out.add(parent)
        return sorted(out)

    def get_call_chain_for_endpoint(self, endpoint, time_minute=None):
        upstream = [(u, 1) for u in self.get_endpoint_upstream(endpoint, time_minute=time_minute)]
        downstream = [(d, 1) for d in self.get_endpoint_downstream(endpoint, time_minute=time_minute)]
        return {
            'upstream': upstream,
            'downstream': downstream,
        }

    def get_endpoint_downstream_in_range(self, endpoint, time_minute):
        range_stats = {}
        example_time_minute = _parse_time_minute(time_minute)
        start_time = example_time_minute - timedelta(minutes=15)
        end_time = example_time_minute + timedelta(minutes=5)
        current_time = start_time
        while current_time <= end_time:
            time_minute_str = _format_time_minute(current_time)
            range_stats[time_minute_str] = self.get_endpoint_downstream(endpoint, time_minute_str)
            current_time += timedelta(minutes=1)
        return range_stats

if __name__ == '__main__':
    explorer = TraceExplorer()

    # example_endpoint = "ts-order-other-service-PUT:/api/v1/orderOtherService/orderOther"
    # example_endpoint = "PUT:/api/v1/orderOtherService/orderOther"
    # example_time_minute = "2024-01-09 09:00:00"
    E = "ts-travel-plan-service-/api/v1/routeplanservice/routePlan/quickestRoute"
    T = "2024-01-09 09:00:00"

    stats_in_range = explorer.get_endpoint_downstream(E, T)
    print(stats_in_range)

    stats_in_range = explorer.get_endpoint_downstream_in_range(E, T)
    print(stats_in_range)
    # print(f"Stats for {E} around {T} (15 time_minutes before and 5 time_minutes after):")
    # for time_minute, stats in stats_in_range.items():
    #     print(f"At {time_minute}: {stats}")