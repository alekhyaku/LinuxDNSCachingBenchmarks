#!/usr/bin/env python3
import argparse
import subprocess
import time
import statistics
import re
import concurrent.futures
import psutil
import os
import random

# Default configuration for rigorous academic testing:
DEFAULT_LOCAL_RESOLVER = "127.0.0.53"  # systemd-resolved stub resolver (change to "127.0.0.1" for dnsmasq)
DEFAULT_QUERY_DOMAIN = "google.com"

# Default test counts:
DEFAULT_TAIL_COUNT = 1000           # Tail latency test
DEFAULT_HIGH_TOTAL = 10000          # High volume test
DEFAULT_HIGH_CONCURRENCY = 100      # High concurrency level
DEFAULT_LOW_COUNT = 100             # Low volume test
DEFAULT_LOW_INTERVAL = 1.0          # 1 second between low-volume queries
DEFAULT_MEMCPU_DURATION = 10        # Mem/CPU sampling for 10 seconds

# Default list of domains if no file is provided:
DEFAULT_DOMAINS = [
    "google.com", "facebook.com", "twitter.com", "amazon.com", "wikipedia.org",
    "youtube.com", "bing.com", "yahoo.com", "reddit.com", "linkedin.com",
    "instagram.com", "office.com", "live.com", "msn.com", "paypal.com",
    "apple.com", "cnn.com", "nytimes.com", "weather.com", "ebay.com"
]

def flush_cache():
    """Flush the systemd-resolved cache."""
    try:
        subprocess.run(["resolvectl", "flush-caches"], check=True)
        time.sleep(3)
        print("[*] Cache flushed.")
    except Exception as e:
        print(f"[!] Failed to flush caches: {e}")

def run_dig(domain=DEFAULT_QUERY_DOMAIN, resolver=DEFAULT_LOCAL_RESOLVER):
    """
    Run a single dig query using the specified resolver and return the query time (in msec) and output.
    We call dig with "+stats" so the output includes query timing.
    """
    try:
        result = subprocess.run(
            ["dig", "@" + resolver, domain, "+stats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()
        query_time_match = re.search(r"Query time:\s+(\d+)\s+msec", output)
        query_time = int(query_time_match.group(1)) if query_time_match else None
        return query_time, output
    except Exception as e:
        print(f"Error running dig for domain {domain}: {e}")
        return None, ""

def tail_latency_test(num_queries, flush=False):
    """Run many queries and compute latency statistics (average, std dev, p95, p99)."""
    if flush:
        flush_cache()
    times = []
    print(f"\nRunning tail latency test with {num_queries} queries...")
    for i in range(num_queries):
        qt, _ = run_dig()
        if qt is not None:
            times.append(qt)
        else:
            print(f"Failed query {i}")
    if times:
        avg = statistics.mean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        quantiles = statistics.quantiles(times, n=100)
        p95 = quantiles[94]
        p99 = quantiles[98]
        print("Tail Latency Test Results:")
        print(f"  Number of queries: {len(times)}")
        print(f"  Average latency: {avg:.2f} msec")
        print(f"  Std Dev: {stdev:.2f} msec")
        print(f"  95th Percentile: {p95} msec")
        print(f"  99th Percentile: {p99} msec")
    else:
        print("No valid query times collected in tail latency test.")

def cache_hit_rate_test():
    """
    Test basic cache hit rate using a single domain.
    Flush the cache; issue one cold query then repeated (warm) queries.
    """
    print("\nRunning single-domain cache hit rate test...")
    flush_cache()
    miss_time, _ = run_dig()
    repeat_times = []
    for _ in range(20):
        qt, _ = run_dig()
        if qt is not None:
            repeat_times.append(qt)
        time.sleep(0.1)
    if miss_time is not None and repeat_times:
        avg_hit = statistics.mean(repeat_times)
        improvement = miss_time - avg_hit
        print(f"Initial (cold) query latency: {miss_time} msec")
        print(f"Average repeated (warm) query latency: {avg_hit:.2f} msec")
        print(f"Cache hit improvement: {improvement:.2f} msec")
    else:
        print("Cache hit rate test failed to collect sufficient data.")

def multi_domain_cache_test(domains, iterations=3):
    """
    Test caching behavior across multiple domains.
    For each domain, flush the cache, issue one cold query, then multiple warm queries.
    Report overall average latencies and improvement.
    """
    print("\nRunning multi-domain cache fill test...")
    flush_cache()
    cold_latencies = []
    warm_latencies = []
    total_domains = len(domains)
    for idx, domain in enumerate(domains, start=1):
        print(f"Testing domain ({idx}/{total_domains}): {domain}")
        cold_time, _ = run_dig(domain=domain)
        if cold_time is None:
            print(f"  [!] Cold query failed for {domain}")
            continue
        cold_latencies.append(cold_time)
        reps = []
        for i in range(iterations):
            qt, _ = run_dig(domain=domain)
            if qt is not None:
                reps.append(qt)
            else:
                print(f"  [!] Warm query {i} failed for {domain}")
            time.sleep(0.05)
        if reps:
            warm_avg = statistics.mean(reps)
            warm_latencies.append(warm_avg)
        else:
            warm_latencies.append(None)
    cold_latencies = [t for t in cold_latencies if t is not None]
    warm_latencies = [t for t in warm_latencies if t is not None]
    if cold_latencies and warm_latencies:
        overall_cold = statistics.mean(cold_latencies)
        overall_warm = statistics.mean(warm_latencies)
        print("\nMulti-Domain Cache Test Results:")
        print(f"  Tested {len(cold_latencies)} domains.")
        print(f"  Average cold query latency: {overall_cold:.2f} msec")
        print(f"  Average warm query latency: {overall_warm:.2f} msec")
        print(f"  Overall improvement: {overall_cold - overall_warm:.2f} msec")
    else:
        print("Insufficient data for multi-domain cache test.")

def eviction_pressure_test(domains, cache_size_guess=1000):
    """
    Test cache eviction by issuing queries to a number of unique domains 
    exceeding the expected cache capacity. After priming the cache with all these domains,
    re-query a subset to measure re-query latency as a proxy for eviction effects.
    """
    print("\nRunning cache eviction pressure test...")
    flush_cache()
    
    # Use a number of domains exceeding the cache capacity guess.
    pressure_domains = domains[:cache_size_guess + 1000]
    print(f"  Using {len(pressure_domains)} domains for pressure test (cache size guess: {cache_size_guess}).")
    
    # Prime the cache.
    for domain in pressure_domains:
        run_dig(domain=domain)
        time.sleep(0.001)  # tiny delay to avoid overwhelming the system
    
    print("  Priming complete. Re-querying the first 10,000 domains to assess eviction...")
    requery_domains = pressure_domains[:1000]
    hit_times = []
    for domain in requery_domains:
        qt, _ = run_dig(domain=domain)
        if qt is not None:
            hit_times.append(qt)
        time.sleep(0.01)
    
    if hit_times:
        avg_hit = statistics.mean(hit_times)
        print(f"  Average latency on re-query: {avg_hit:.2f} msec")
        print(f"  (Higher latency indicates more evictions.)")
    else:
        print("  No valid data collected during eviction pressure test.")

def run_random_sample_tests(loaded_domains, sample_sizes=[1000, 10000, 100000, 1000000]):
    """
    For various sample sizes, select a random sample from the loaded domains and run the multi-domain cache test.
    """
    print("\nRunning multi-domain cache tests for various random sample sizes:")
    for s in sample_sizes:
        if len(loaded_domains) >= s:
            sample = random.sample(loaded_domains, s)
            print(f"\n--- Testing with random sample of {s} domains ---")
            multi_domain_cache_test(sample, iterations=3)
        else:
            print(f"\nInsufficient domains to sample {s} domains (only {len(loaded_domains)} available).")

def ttl_handling_test():
    """Test TTL handling by parsing TTL values from a DNS response."""
    print("\nRunning TTL handling test...")
    _, output = run_dig()
    ttl_values = re.findall(r'\s+(\d+)\s+IN\s', output)
    if ttl_values:
        ttl_values = list(map(int, ttl_values))
        print(f"Extracted TTL values: {ttl_values}")
        print(f"Minimum TTL: {min(ttl_values)} seconds")
        print(f"Maximum TTL: {max(ttl_values)} seconds")
    else:
        print("Unable to parse TTL values from dig output.")

def high_volume_test(total_requests, concurrency, flush=False):
    """Issue many queries concurrently and record latency statistics."""
    if flush:
        flush_cache()
    print(f"\nRunning high volume test with {total_requests} queries at concurrency level {concurrency}...")
    query_times = []
    def worker():
        qt, _ = run_dig()
        if qt is not None:
            query_times.append(qt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(total_requests)]
        concurrent.futures.wait(futures)
    if query_times:
        avg = statistics.mean(query_times)
        p99 = statistics.quantiles(query_times, n=100)[98]
        print("High Volume Test Results:")
        print(f"  Total queries executed: {len(query_times)}")
        print(f"  Average latency: {avg:.2f} msec")
        print(f"  99th Percentile latency: {p99} msec")
    else:
        print("No valid query times collected in high volume test.")

def low_volume_test(num_queries, interval, flush=False):
    """Issue queries at a low, regular interval and report average latency."""
    if flush:
        flush_cache()
    print(f"\nRunning low volume test with {num_queries} queries, one every {interval} seconds...")
    query_times = []
    for i in range(num_queries):
        qt, _ = run_dig()
        if qt is not None:
            query_times.append(qt)
        else:
            print(f"Failed query {i}")
        time.sleep(interval)
    if query_times:
        avg = statistics.mean(query_times)
        print("Low Volume Test Results:")
        print(f"  Total queries executed: {len(query_times)}")
        print(f"  Average latency: {avg:.2f} msec")
    else:
        print("No valid query times collected in low volume test.")

def memory_cpu_utilization_test(duration):
    """Monitor memory and CPU usage of systemd-resolved over a specified duration (in seconds)."""
    print(f"\nRunning memory and CPU utilization test for {duration} seconds...")
    proc = None
    for p in psutil.process_iter(['name']):
        try:
            if p.info['name'] == "systemd-resolved":
                proc = p
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not proc:
        print("systemd-resolved process not found!")
        return
    mem_samples = []
    cpu_samples = []
    for i in range(duration):
        try:
            mem = proc.memory_info().rss / (1024 * 1024)  # in MB
            cpu = proc.cpu_percent(interval=1.0)
            mem_samples.append(mem)
            cpu_samples.append(cpu)
            print(f"Sample {i+1}: Memory = {mem:.2f} MB, CPU = {cpu:.2f}%")
        except psutil.NoSuchProcess:
            print("systemd-resolved process ended unexpectedly.")
            break
    if mem_samples and cpu_samples:
        print(f"Average Memory: {statistics.mean(mem_samples):.2f} MB")
        print(f"Average CPU: {statistics.mean(cpu_samples):.2f}%")
    else:
        print("No samples collected.")

def fallback_test():
    """
    Simulate DNS-over-TLS failure to trigger fallback to unencrypted UDP.
    
    This function temporarily sets the upstream DNS server on the primary interface
    to an unreachable IP (203.0.113.1) so that encrypted queries fail. If systemd-resolved 
    is configured with fallback (i.e. it can revert to using unencrypted UDP), that 
    fallback mechanism should be triggered.
    
    After the test, the DNS configuration is reverted back to the known good upstream.
    """
    fallback_interface = "enp0s3"  # Replace with your actual network interface name
    print("\nRunning fallback test (simulate encrypted failure)...")
    try:
        print(f"[*] Temporarily setting DNS server on {fallback_interface} to 203.0.113.1 to simulate failure.")
        subprocess.run(["resolvectl", "dns", fallback_interface, "203.0.113.1"], check=True)
        flush_cache()
        time.sleep(2)
        
        qt, output = run_dig()
        print("\nTest query result when primary (encrypted) fails:")
        print(output)
        
        status_output = subprocess.run(["resolvectl", "status"], stdout=subprocess.PIPE, text=True).stdout
        print("\nResolver status after simulated failure:")
        print(status_output)
    except Exception as e:
        print(f"Error during fallback test: {e}")
    finally:
        print(f"[*] Reverting DNS server on {fallback_interface} back to 8.8.8.8.")
        subprocess.run(["resolvectl", "dns", fallback_interface, "8.8.8.8"], check=True)
        flush_cache()
        time.sleep(2)

def load_domains_from_file(filepath):
    """Load a list of domains from a file (one domain per line)."""
    if not os.path.isfile(filepath):
        print(f"Domain file {filepath} not found. Using default domain list.")
        return DEFAULT_DOMAINS
    with open(filepath, "r") as f:
        domains = [line.strip() for line in f if line.strip()]
    if not domains:
        print(f"Domain file {filepath} is empty. Using default domain list.")
        return DEFAULT_DOMAINS
    return domains

def main():
    parser = argparse.ArgumentParser(
        description="Rigorous Benchmarking for systemd-resolved performance testing."
    )
    parser.add_argument("--mode", choices=["unencrypted", "encrypted", "fallback"], required=True,
                        help="DNS mode to test: unencrypted, encrypted (DoT) or encryption with fallback.")
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL_COUNT,
                        help="Number of queries for tail latency test (default: 1000).")
    parser.add_argument("--high_total", type=int, default=DEFAULT_HIGH_TOTAL,
                        help="Total queries for high volume test (default: 10000).")
    parser.add_argument("--high_concurrency", type=int, default=DEFAULT_HIGH_CONCURRENCY,
                        help="Concurrency for high volume test (default: 100).")
    parser.add_argument("--low_count", type=int, default=DEFAULT_LOW_COUNT,
                        help="Number of queries for low volume test (default: 100).")
    parser.add_argument("--low_interval", type=float, default=DEFAULT_LOW_INTERVAL,
                        help="Interval (in seconds) for low volume test (default: 1.0).")
    parser.add_argument("--memcpu_duration", type=int, default=DEFAULT_MEMCPU_DURATION,
                        help="Duration (in seconds) for memory/CPU test (default: 10).")
    parser.add_argument("--domains_file", type=str, default="",
                        help="Path to a file containing a list of domains (one per line) for multi-domain cache testing.")
    parser.add_argument("--flush", action="store_true",
                        help="Flush DNS cache before tests that need a cold start.")
    parser.add_argument("--fallback_test", action="store_true",
                        help="Run the fallback test (simulate encrypted failure and expect fallback to UDP).")

    args = parser.parse_args()
    
    print(f"Starting tests in {args.mode.upper()} mode using 8.8.8.8 as upstream (via stub resolver {DEFAULT_LOCAL_RESOLVER}).")
    
    tail_latency_test(args.tail, flush=args.flush)
    cache_hit_rate_test()
    ttl_handling_test()
    
    loaded_domains = load_domains_from_file(args.domains_file) if args.domains_file else DEFAULT_DOMAINS
    
    # Run multi-domain cache test on a moderate subset (e.g., first 100 domains) for quick evaluation.
    multi_domain_cache_test(loaded_domains[:100], iterations=3)
    
    high_volume_test(args.high_total, args.high_concurrency, flush=args.flush)
    low_volume_test(args.low_count, args.low_interval, flush=args.flush)
    memory_cpu_utilization_test(args.memcpu_duration)
    
    if args.fallback_test:
        fallback_test()
    
    # Run eviction pressure test if the loaded domain file is large (at least 100,000 entries).
    if len(loaded_domains) >= 100000:
        eviction_pressure_test(loaded_domains, cache_size_guess=1000)
    
    # Automatically run multi-domain cache tests for various random sample sizes if possible.
    sample_sizes = [2000]
    print("\n=== Running Random Sample Tests for Various Sample Sizes ===")
    for s in sample_sizes:
        if len(loaded_domains) >= s:
            sample = random.sample(loaded_domains, s)
            print(f"\n--- Testing with random sample of {s} domains ---")
            multi_domain_cache_test(sample, iterations=3)
        else:
            print(f"\nInsufficient domains to sample {s} domains (available: {len(loaded_domains)}).")
    
if __name__ == '__main__':
    main()