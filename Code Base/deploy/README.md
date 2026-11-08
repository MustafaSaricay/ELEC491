# Putting the interface online (Cloudflare Tunnel)

The simulator keeps running on your own computer. A Cloudflare Tunnel publishes it at a hostname on your domain, and Cloudflare Access puts an email login in front of it.

```
browser ──https──► Cloudflare (Access login) ──tunnel──► cloudflared on your PC ──► 127.0.0.1:8765 (parksim ui)
```

Nothing listens on a public port. The interface only accepts connections from the same machine (`127.0.0.1`), and the tunnel is an outgoing connection.

## 1. Create the tunnel (Cloudflare dashboard)

1. Go to **Zero Trust → Networks → Tunnels → Create a tunnel**. Choose the *Cloudflared* type and name it `parksim`.
2. On the "Install connector" page, **copy the token**: the long string after `--token` in the command it shows. You don't need to run that command, because `cloudflared` is already installed.
3. **Public hostname:** pick a subdomain such as `parksim` on your domain, set the service to **HTTP** and the URL to `localhost:8765`. This adds the DNS record automatically and doesn't affect the rest of your site.

## 2. Add a login (do this before sharing the link)

1. Go to **Zero Trust → Access → Applications → Add an application → Self-hosted**.
2. Set the application domain to the same hostname, e.g. `parksim.yourdomain.com`.
3. Add a policy with action **Allow**, include **Emails** (your team's addresses), or **Emails ending in** `@ku.edu.tr`.
4. Use the default login method, a **one-time PIN** sent by email.

## 3. Store the token (only on your machine; never commit it)

```bash
mkdir -p ~/.config/parksim
printf 'TUNNEL_TOKEN=%s\n' 'PASTE-THE-TOKEN-HERE' > ~/.config/parksim/tunnel.env
chmod 600 ~/.config/parksim/tunnel.env
```

## 4a. Run it by hand (two terminals)

```bash
cd ~/Documents/Courses/ELEC491/"Code Base"
.venv/bin/python -m parksim ui --no-browser --workers 4 --max-runs 100 --max-sims 400
```
```bash
set -a; . ~/.config/parksim/tunnel.env; set +a
cloudflared tunnel --no-autoupdate run
```

## 4b. Or as background services (keep running without a terminal)

```bash
mkdir -p ~/.config/systemd/user
cp ~/Documents/Courses/ELEC491/"Code Base"/deploy/parksim-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now parksim-ui.service parksim-tunnel.service
systemctl --user status parksim-ui parksim-tunnel        # check both are "active (running)"
journalctl --user -u parksim-tunnel -f                   # tunnel log (Ctrl+C to leave)
```

- **Stop:** `systemctl --user stop parksim-tunnel parksim-ui`. **Disable on boot:** `systemctl --user disable parksim-tunnel parksim-ui`.
- **Keep running when logged out:** user services stop when you log out unless you run `loginctl enable-linger $USER` once.
- **Keep the machine awake:** Fedora suspends after a while by default, and the site goes down while it sleeps. Change this in Settings → Power, or accept it.
- **After changing the code:** run `systemctl --user restart parksim-ui`.

## Limits that protect your machine

These are set in `parksim-ui.service` and can be changed there:

| Flag | Default in the service | Meaning |
|---|---|---|
| `--workers 4` | 4 processes | CPU used by Monte Carlo and comparisons (your machine has more cores) |
| `--max-runs 100` | 100 | runs per Monte Carlo job |
| `--max-sims 400` | 400 | variants × runs per comparison |
| `--keep-jobs` | 20 | finished results kept in memory (each single-run result is about 1 MB) |

The server also clamps settings that could make a single job run for hours: `mission.max_time` ≤ 300 s, `sim.dt` ≥ 2 ms, planner expansions ≤ 20,000, and so on (`CLAMPS` in `parksim/ui/server.py`). Only one job runs at a time, and everyone shares it.
