
const API_BASE = "http://127.0.0.1:8000/api/v1";

async function updateDashboard() {
    try {
        // 1. Fetch Health
        const health = await fetch(`${API_BASE}/health`).then(r => r.json());
        const statusBadge = document.getElementById('status-badge');
        statusBadge.textContent = health.status;
        statusBadge.style.color = health.status === 'ACTIVE' ? '#00ff95' : '#ff3e3e';

        // 2. Fetch Equity
        const equity = await fetch(`${API_BASE}/equity`).then(r => r.json());
        document.getElementById('equity-val').textContent = `$${equity.current.toFixed(2)}`;
        document.getElementById('baseline-val').textContent = `$${equity.baseline.toFixed(2)}`;

        const perf = ((equity.current - equity.baseline) / equity.baseline) * 100;
        const perfEl = document.getElementById('equity-pct');
        perfEl.textContent = `${perf >= 0 ? '+' : ''}${perf.toFixed(2)}%`;
        perfEl.className = perf >= 0 ? 'pnl-pos' : 'pnl-neg';

        // 3. Fetch Stats
        const stats = await fetch(`${API_BASE}/stats`).then(r => r.json());
        document.getElementById('win-rate').textContent = `${stats.win_rate.toFixed(1)}%`;

        // 4. Fetch Positions
        const positions = await fetch(`${API_BASE}/positions`).then(r => r.json());
        document.getElementById('pos-count').textContent = positions.length;
        const posList = document.getElementById('positions-list');
        posList.innerHTML = positions.map(p => `
            <div class="pos-item">
                <div>
                    <strong>${p.symbol}</strong><br>
                    <small>${p.side} | ${p.size.toFixed(4)}</small>
                </div>
                <div class="${p.pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">
                    ${p.pnl >= 0 ? '+' : ''}${p.pnl.toFixed(2)}
                </div>
            </div>
        `).join('');

        // 5. Fetch Signals
        const signals = await fetch(`${API_BASE}/signals`).then(r => r.json());
        const signalStream = document.getElementById('signal-stream');
        signalStream.innerHTML = signals.map(s => `<div>${s}</div>`).join('');

        document.getElementById('last-update').textContent = `LAST SYNC: ${new Date().toLocaleTimeString()}`;

    } catch (err) {
        console.error("Nexus Sync Error:", err);
        document.getElementById('status-badge').textContent = "CONNECTION ERROR";
    }
}

setInterval(updateDashboard, 3000);
updateDashboard();
