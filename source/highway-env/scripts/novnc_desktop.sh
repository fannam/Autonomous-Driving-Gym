#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"

DISPLAY_NUM="${DISPLAY_NUM:-1}"
GEOMETRY="${GEOMETRY:-1920x1080}"
DEPTH="${DEPTH:-24}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
NOVNC_WEB_ROOT="${NOVNC_WEB_ROOT:-/usr/share/novnc}"
VNC_HOST="${VNC_HOST:-localhost}"
VNC_PORT=$((5900 + DISPLAY_NUM))
PID_DIR="${HOME}/.vnc"
WEBSOCKIFY_PIDFILE="${PID_DIR}/websockify-${DISPLAY_NUM}.pid"
WEBSOCKIFY_LOG="${PID_DIR}/websockify-${DISPLAY_NUM}.log"
SSH_USER_HOST="${SSH_USER_HOST:-${USER}@<VM_IP>}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|status|restart]

Environment variables:
  DISPLAY_NUM      VNC display number (default: 1)
  GEOMETRY         Desktop resolution (default: 1920x1080)
  DEPTH            Color depth (default: 24)
  NOVNC_PORT       noVNC/websockify port on VM (default: 6080)
  NOVNC_WEB_ROOT   noVNC static web root (default: /usr/share/novnc)
  SSH_USER_HOST    Value shown in tunnel hint (default: ${USER}@<VM_IP>)
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing command: $cmd" >&2
    exit 1
  fi
}

is_vnc_running() {
  if vncserver -list 2>/dev/null | grep -Eq "^[[:space:]]*:${DISPLAY_NUM}[[:space:]]"; then
    return 0
  fi

  # Fallback for cases where `vncserver -list` format differs.
  local pid_file
  for pid_file in "${HOME}"/.vnc/*":${DISPLAY_NUM}.pid"; do
    if [[ -f "$pid_file" ]]; then
      local pid
      pid="$(cat "$pid_file" 2>/dev/null || true)"
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        return 0
      fi
    fi
  done

  return 1
}

is_websockify_running() {
  if [[ ! -f "$WEBSOCKIFY_PIDFILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$WEBSOCKIFY_PIDFILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_vnc() {
  if is_vnc_running; then
    echo "VNC display :${DISPLAY_NUM} already running."
    return
  fi
  echo "Starting VNC display :${DISPLAY_NUM} (${GEOMETRY}, depth ${DEPTH})..."
  local output
  if output="$(vncserver ":${DISPLAY_NUM}" -localhost yes -geometry "$GEOMETRY" -depth "$DEPTH" 2>&1)"; then
    echo "$output"
    return
  fi

  if grep -qi "already running for display :${DISPLAY_NUM}" <<<"$output"; then
    echo "VNC display :${DISPLAY_NUM} already running."
    return
  fi

  echo "$output" >&2
  exit 1
}

start_websockify() {
  mkdir -p "$PID_DIR"
  if [[ ! -d "$NOVNC_WEB_ROOT" ]]; then
    echo "noVNC web root not found: $NOVNC_WEB_ROOT" >&2
    echo "Set NOVNC_WEB_ROOT or install noVNC." >&2
    exit 1
  fi

  if is_websockify_running; then
    echo "websockify already running (pid $(cat "$WEBSOCKIFY_PIDFILE"))."
    return
  fi

  rm -f "$WEBSOCKIFY_PIDFILE"
  echo "Starting websockify on :${NOVNC_PORT} -> ${VNC_HOST}:${VNC_PORT} ..."
  nohup websockify --web="$NOVNC_WEB_ROOT" "$NOVNC_PORT" "${VNC_HOST}:${VNC_PORT}" \
    >"$WEBSOCKIFY_LOG" 2>&1 &
  echo "$!" >"$WEBSOCKIFY_PIDFILE"
  sleep 0.3

  if ! is_websockify_running; then
    echo "websockify failed to start. See log: $WEBSOCKIFY_LOG" >&2
    exit 1
  fi
}

print_access_hint() {
  cat <<EOF
Ready.

1) On local machine, create tunnel:
   ssh -L ${NOVNC_PORT}:localhost:${NOVNC_PORT} ${SSH_USER_HOST}

2) Open browser on local machine:
   http://localhost:${NOVNC_PORT}/vnc.html

Status:
  VNC display :${DISPLAY_NUM} -> localhost:${VNC_PORT}
  websockify  :${NOVNC_PORT}
  log         ${WEBSOCKIFY_LOG}
EOF
}

stop_websockify() {
  if is_websockify_running; then
    local pid
    pid="$(cat "$WEBSOCKIFY_PIDFILE")"
    echo "Stopping websockify (pid ${pid})..."
    kill "$pid" || true
    sleep 0.2
  else
    echo "websockify is not running."
  fi
  rm -f "$WEBSOCKIFY_PIDFILE"
}

stop_vnc() {
  if is_vnc_running; then
    echo "Stopping VNC display :${DISPLAY_NUM}..."
    vncserver -kill ":${DISPLAY_NUM}" || true
  else
    echo "VNC display :${DISPLAY_NUM} is not running."
  fi
}

status() {
  echo "Display :${DISPLAY_NUM}, VNC port ${VNC_PORT}, noVNC port ${NOVNC_PORT}"
  if is_vnc_running; then
    echo "VNC: running"
  else
    echo "VNC: stopped"
  fi

  if is_websockify_running; then
    echo "websockify: running (pid $(cat "$WEBSOCKIFY_PIDFILE"))"
  else
    echo "websockify: stopped"
  fi
}

start() {
  require_cmd vncserver
  require_cmd websockify
  start_vnc
  start_websockify
  print_access_hint
}

stop() {
  stop_websockify
  stop_vnc
}

restart() {
  stop || true
  start
}

case "$ACTION" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  restart) restart ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage
    exit 1
    ;;
esac
