module.exports = {
    apps: [{
        name: "sovereign-bot",
        script: "./run_v17_async.py",
        interpreter: "./venv/bin/python3",
        env: {
            NODE_ENV: "production",
            LIVE_MODE: "true"
        },
        autorestart: true,
        watch: false,
        max_memory_restart: '800M',
        exp_backoff_restart_delay: 10000
    },
    {
        name: "sovereign-dashboard",
        script: "./venv/bin/python3",
        args: "-m streamlit run dashboard_antigravity.py --server.port 8501 --server.headless true",
        interpreter: "none",
        autorestart: true,
        watch: false,
        max_memory_restart: '1G'
    },
    {
        name: "sovereign-sentinel",
        script: "./evolution_sentinel.py",
        interpreter: "./venv/bin/python3",
        autorestart: true,
        watch: false
    }
    ]
};
