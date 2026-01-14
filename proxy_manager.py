
import requests
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class ProxyManager:
    def __init__(self, proxy_list=None, skip_init=False):
        self.proxies = proxy_list or []
        self.failed_proxies = {} 
        self.max_failures = 2
        self.refresh_threshold = 3
        self.last_refresh = 0
        self.refresh_cooldown = 300 # 5 minutes
        print("🪐 [PROXY] Manager Initialized.")
        
        if not self.proxies and not skip_init:
            self.refresh_proxies()
        
    def get_proxy(self, allow_refresh=True):
        healthy = [p for p in self.proxies if self.failed_proxies.get(p, 0) < self.max_failures]
        
        # Phase 27 Dashboard Guard: Don't block the UI with a full refresh if we are already testing.
        if len(healthy) < self.refresh_threshold and allow_refresh:
            now = time.time()
            if now - self.last_refresh > self.refresh_cooldown:
                print("🔄 [PROXY] Healthy list low. Triggering refresh...")
                self.refresh_proxies()
                self.last_refresh = now
                healthy = [p for p in self.proxies if self.failed_proxies.get(p, 0) < self.max_failures]
        
        if not healthy:
            if self.proxies:
                proxy = random.choice(self.proxies)
                return {
                    "http": f"http://{proxy}",
                    "https": f"http://{proxy}"
                }
            return None
        proxy = random.choice(healthy)
        return {
            "http": f"http://{proxy}",
            "https": f"http://{proxy}"
        }

    def report_failure(self, proxy_dict):
        if not proxy_dict: return
        proxy_str = proxy_dict['http'].replace('http://', '')
        self.failed_proxies[proxy_str] = self.failed_proxies.get(proxy_str, 0) + 1
        print(f"⚠️ [PROXY] Failure: {proxy_str} ({self.failed_proxies[proxy_str]})")

    def report_success(self, proxy_dict):
        if not proxy_dict: return
        proxy_str = proxy_dict['http'].replace('http://', '')
        if proxy_str in self.failed_proxies:
            del self.failed_proxies[proxy_str]

    def test_proxies(self, test_url="https://api.binance.com/api/v3/ping"):
        if not self.proxies: return 0
        print(f"🛰️ [PROXY] Testing {len(self.proxies)} proxies concurrently...")
        
        valid = []
        def check_one(p):
            p_dict = {"http": f"http://{p}", "https": f"http://{p}"}
            try:
                # Increased timeout to 15s for slow free proxies
                resp = requests.get(test_url, proxies=p_dict, timeout=15)
                if resp.status_code == 200:
                    return p
            except:
                pass
            return None

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(check_one, p) for p in self.proxies]
            for future in as_completed(futures):
                res = future.result()
                if res and res not in valid:
                    valid.append(res)
        
        if valid:
            self.proxies = valid
            print(f"✅ [PROXY] Validated {len(valid)} healthy paths.")
        else:
            print("⚠️ [PROXY] No proxies passed validation. Retaining current pool for emergency fallback.")
        return len(valid)

    def refresh_proxies(self):
        sources = [
            "https://proxylist.geonode.com/api/proxy-list?limit=100&page=1&sort_by=lastChecked&sort_type=desc&protocols=http%2Csocks4%2Csocks5",
            "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
            "https://spys.me/proxy.txt",
            "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
            "https://raw.githubusercontent.com/shiftytr/proxy-list/master/proxy.txt",
            "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
            "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt",
            "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/main/http.txt",
            "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/http/data.txt"
        ]
        
        all_new = []
        for url in sources:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    if "geonode" in url:
                        data = resp.json().get('data', [])
                        all_new += [f"{p['ip']}:{p['port']}" for p in data]
                    elif "spys.me" in url:
                        lines = resp.text.split('\n')
                        all_new += [l.split(' ')[0] for l in lines if ':' in l and l[0].isdigit()]
                    else:
                        all_new += resp.text.strip().split('\n')
            except:
                continue
        
        if all_new:
            all_new = [p.strip() for p in all_new if p and ":" in p]
            random.shuffle(all_new)
            all_new = all_new[:100]
            
            self.proxies = list(set(self.proxies + all_new))
            print(f"✅ [PROXY] Multiplexed {len(all_new)} proxies. Total potential: {len(self.proxies)}")
            self.test_proxies()
        else:
            print("❌ [PROXY] Source Fetch Failure.")

if __name__ == "__main__":
    pm = ProxyManager()
    pm.get_proxy()
    print(f"Pool size: {len(pm.proxies)}")
