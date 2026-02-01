import json
import re
from datetime import datetime, timedelta


_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")


def _parse_time_minute(value: str) -> datetime:
    s = str(value or "").strip()
    if not s:
        raise ValueError("time_minute is empty")
    # Accept either 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD HH:MM:SS'.
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # A few LLMs sometimes pass ISO-like strings; try a minimal normalize.
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
    # Lowercase, split on non-alnum, drop generic tokens like 'service'.
    tokens = re.split(r"[^a-zA-Z0-9_]+", str(s or "").strip().lower())
    drop = {"service", "svc", "api"}
    tokens = [t for t in tokens if t and t not in drop]
    return " ".join(tokens)


def _suffix_key(s: str) -> str:
    # For keys like 'checkoutservice-hipstershop.PaymentService/Charge',
    # suffix is 'hipstershop.PaymentService/Charge'.
    ss = str(s or "").strip()
    if "-" in ss:
        return ss.split("-", 1)[1]
    return ss


class MetricExplorer:
    def __init__(self):
        stats_file="data/metric/endpoint_stats.json"
        self.aggregated_stats = self.load_data(stats_file)

        # Build lightweight lookup indexes to tolerate endpoint naming variants.
        self._keys = list(self.aggregated_stats.keys())
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
        if ep in self.aggregated_stats:
            return [ep]

        # 1) Exact normalized match.
        nk = _norm_endpoint(ep)
        keys = self._norm_to_keys.get(nk, [])
        if keys:
            return sorted(set(keys))

        # 2) Suffix match (ignoring the first segment before '-').
        sk = _suffix_key(ep)
        keys = self._suffix_to_keys.get(sk, [])
        if keys:
            return sorted(set(keys))

        # 3) Heuristic: if query looks like 'payment-hipstershop.PaymentService/Charge'
        # but data has 'checkoutservice-hipstershop.PaymentService/Charge', match on
        # the part after the first '-' if possible.
        if "-" in ep:
            tail = ep.split("-", 1)[1]
            keys = self._suffix_to_keys.get(tail, [])
            if keys:
                return sorted(set(keys))

        return []

    def _merge_stats(self, stats_list: list[dict]) -> dict:
        if not stats_list:
            return {}

        calls_total = 0
        sum_duration = 0.0
        sum_error = 0.0
        sum_success = 0.0
        sum_timeout = 0.0

        for s in stats_list:
            if not isinstance(s, dict):
                continue
            calls = int(s.get("calls", 0) or 0)
            calls_total += calls
            if calls > 0:
                sum_duration += float(s.get("average_duration", 0) or 0) * calls
                sum_error += float(s.get("error_rate", 0) or 0) * calls
                sum_success += float(s.get("success_rate", 0) or 0) * calls
                sum_timeout += float(s.get("timeout_rate", 0) or 0) * calls

        if calls_total <= 0:
            return {
                "calls": 0,
                "success_rate": 0,
                "error_rate": 0,
                "average_duration": 0,
                "timeout_rate": 0,
            }

        return {
            "calls": calls_total,
            "success_rate": sum_success / calls_total,
            "error_rate": sum_error / calls_total,
            "average_duration": sum_duration / calls_total,
            "timeout_rate": sum_timeout / calls_total,
        }

    def query_endpoint_stats(self, endpoint, time_minute):
        keys = self._resolve_endpoint_keys(endpoint)
        if not keys:
            return {}

        # Normalize time key to match our stored minute format.
        try:
            dt = _parse_time_minute(time_minute)
            t = _format_time_minute(dt)
        except Exception:
            t = str(time_minute or "").strip()

        if len(keys) == 1:
            endpoint_data = self.aggregated_stats.get(keys[0], {})
            return endpoint_data.get(t, {})

        stats = []
        for k in keys:
            endpoint_data = self.aggregated_stats.get(k, {})
            stats.append(endpoint_data.get(t, {}))
        return self._merge_stats(stats)

    def query_endpoint_stats_in_range(self, endpoint, time_minute):
        range_stats = {}
        example_time_minute = _parse_time_minute(time_minute)
        start_time = example_time_minute - timedelta(minutes=15)
        end_time = example_time_minute + timedelta(minutes=5)
        current_time = start_time
        keys = self._resolve_endpoint_keys(endpoint)
        while current_time <= end_time:
            time_minute_str = _format_time_minute(current_time)
            if not keys:
                range_stats[time_minute_str] = {
                    'calls': 0,
                    'success_rate': 0,
                    'error_rate': 0,
                    'average_duration': 0,
                    'timeout_rate': 0,
                }
            elif len(keys) == 1:
                endpoint_data = self.aggregated_stats.get(keys[0], {})
                range_stats[time_minute_str] = endpoint_data.get(
                    time_minute_str,
                    {'calls': 0, 'success_rate': 0, 'error_rate': 0, 'average_duration': 0, 'timeout_rate': 0},
                )
            else:
                stats = []
                for k in keys:
                    endpoint_data = self.aggregated_stats.get(k, {})
                    stats.append(
                        endpoint_data.get(
                            time_minute_str,
                            {'calls': 0, 'success_rate': 0, 'error_rate': 0, 'average_duration': 0, 'timeout_rate': 0},
                        )
                    )
                range_stats[time_minute_str] = self._merge_stats(stats)
            current_time += timedelta(minutes=1)
        return range_stats

if __name__ == '__main__':
    explorer = MetricExplorer()

    # example_endpoint = "ts-order-other-service-PUT:/api/v1/orderOtherService/orderOther"
    # example_endpoint = "PUT:/api/v1/orderOtherService/orderOther"
    # example_time_minute = "2024-01-09 09:00:00"
    E = "ts-travel-plan-service-/api/v1/routeplanservice/routePlan/quickestRoute"
    T = "2024-01-09 09:00:00"

    stats_in_range = explorer.query_endpoint_stats(E, T)
    print(stats_in_range)

    stats_in_range = explorer.query_endpoint_stats_in_range(E, T)
    print(f"Stats for {E} around {T} (15 time_minutes before and 5 time_minutes after):")
    for time_minute, stats in stats_in_range.items():
        print(f"At {time_minute}: {stats}")
