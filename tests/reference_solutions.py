"""Reference solutions used to generate/verify expected outputs in config/coding_problems.yaml."""
from collections import Counter, defaultdict, deque
import statistics


def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []


def valid_parentheses(s):
    pairs, st = {")": "(", "]": "[", "}": "{"}, []
    for ch in s:
        if ch in "([{":
            st.append(ch)
        elif not st or st.pop() != pairs[ch]:
            return False
    return not st


def merge_intervals(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def top_k_frequent(nums, k):
    return [x for x, _ in Counter(nums).most_common(k)]


def length_of_longest_substring(s):
    last, start, best = {}, 0, 0
    for i, ch in enumerate(s):
        if last.get(ch, -1) >= start:
            start = last[ch] + 1
        last[ch] = i
        best = max(best, i - start + 1)
    return best


def num_islands(grid):
    g = [list(r) for r in grid]
    R, C, n = len(g), len(g[0]) if g else 0, 0
    for r in range(R):
        for c in range(C):
            if g[r][c] == "1":
                n += 1
                q = deque([(r, c)]); g[r][c] = "0"
                while q:
                    a, b = q.popleft()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < R and 0 <= y < C and g[x][y] == "1":
                            g[x][y] = "0"; q.append((x, y))
    return n


def brute_force_ips(logs, threshold, window):
    fails = defaultdict(list)
    for ts, ip, status in sorted(logs):
        if status == "FAIL":
            fails[ip].append(ts)
    out = []
    for ip, ts in fails.items():
        q = deque()
        for t in ts:
            q.append(t)
            while q[0] <= t - window:
                q.popleft()
            if len(q) >= threshold:
                out.append(ip); break
    return sorted(out)


def rbac_can_access(role_parents, role_perms, user_roles, permission):
    seen, stack = set(), list(user_roles)
    while stack:
        r = stack.pop()
        if r in seen:
            continue
        seen.add(r)
        if permission in role_perms.get(r, []):
            return True
        stack.extend(role_parents.get(r, []))
    return False


def rolling_zscore_anomalies(values, window, threshold):
    out = []
    for i in range(window, len(values)):
        prev = values[i - window:i]
        sd = statistics.pstdev(prev)
        if sd > 0 and abs(values[i] - statistics.mean(prev)) / sd > threshold:
            out.append(i)
    return out
