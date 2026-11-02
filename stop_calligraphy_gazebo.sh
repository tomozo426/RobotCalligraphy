#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
world_file="$workspace_root/calligraphy_world.sdf"
gui_config="$workspace_root/build/gui_texture/gui.config"
dry_run=0
force=0

usage() {
    printf 'Usage: %s [--dry-run] [--force]\n' "${0##*/}"
    printf '  --dry-run  List matching processes without signaling them.\n'
    printf '  --force    Send SIGKILL if processes remain after SIGINT and SIGTERM.\n'
}

while (($#)); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --force) force=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done

declare -A selected=()

add_process_tree() {
    local pid="$1"
    local children_file="/proc/$pid/task/$pid/children"
    local children=""
    local child

    [[ "$pid" =~ ^[0-9]+$ && -d "/proc/$pid" ]] || return 0
    [[ -n "${selected[$pid]+present}" ]] && return 0
    selected["$pid"]=1

    [[ -r "$children_file" ]] || return 0
    IFS= read -r children < "$children_file" || true
    for child in $children; do
        add_process_tree "$child"
    done
}

for proc_dir in /proc/[0-9]*; do
    pid="${proc_dir##*/}"
    [[ -r "$proc_dir/cmdline" && -e "$proc_dir/cwd" ]] || continue
    command_line="$(tr '\0' ' ' < "$proc_dir/cmdline" 2>/dev/null || true)"
    process_cwd="$(readlink "$proc_dir/cwd" 2>/dev/null || true)"

    is_launch=0
    if [[ "$process_cwd" == "$workspace_root" &&
          "$command_line" == *"ros2 launch"* &&
          "$command_line" == *"ros2_calligraphy.launch.py"* ]]; then
        is_launch=1
    fi

    is_gazebo=0
    if [[ "$command_line" == *"$world_file"* ||
          "$command_line" == *"$gui_config"* ]]; then
        is_gazebo=1
    elif [[ "$process_cwd" == "$workspace_root" &&
            ( "$command_line" == *"ign gazebo server"* ||
              "$command_line" == *"ign gazebo gui"* ) ]]; then
        is_gazebo=1
    fi

    if ((is_launch || is_gazebo)); then
        add_process_tree "$pid"
    fi
done

if ((${#selected[@]} == 0)); then
    printf 'No calligraphy Gazebo or launch processes found.\n'
    exit 0
fi

mapfile -t pids < <(printf '%s\n' "${!selected[@]}" | sort -n)
printf 'Matching calligraphy Gazebo process tree(s):\n'
ps -o pid=,ppid=,stat=,args= -p "$(IFS=,; printf '%s' "${pids[*]}")"

if ((dry_run)); then
    printf 'Dry run: no signals sent.\n'
    exit 0
fi

send_signal() {
    local signal="$1"
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -s "$signal" "$pid" 2>/dev/null || true
        fi
    done
}

remaining_pids() {
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            printf '%s\n' "$pid"
        fi
    done
}

wait_for_exit() {
    local attempts="$1"
    local attempt
    for ((attempt = 0; attempt < attempts; attempt++)); do
        [[ -z "$(remaining_pids)" ]] && return 0
        sleep 0.1
    done
    [[ -z "$(remaining_pids)" ]]
}

send_signal INT
if ! wait_for_exit 30; then
    printf 'Some processes remain; sending SIGTERM.\n'
    send_signal TERM
    wait_for_exit 20 || true
fi

survivors="$(remaining_pids)"
if [[ -n "$survivors" && "$force" -eq 1 ]]; then
    printf 'Some processes remain; --force sends SIGKILL.\n'
    send_signal KILL
    wait_for_exit 10 || true
    survivors="$(remaining_pids)"
fi

if [[ -n "$survivors" ]]; then
    printf 'Processes did not exit: %s\n' "${survivors//$'\n'/ }" >&2
    printf 'Review them, then rerun with --force if appropriate.\n' >&2
    exit 1
fi

printf 'All matching calligraphy Gazebo processes exited.\n'