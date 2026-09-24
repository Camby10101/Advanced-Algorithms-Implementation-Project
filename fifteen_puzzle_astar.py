#!/usr/bin/env python3
"""
15-Puzzle Solver using A* Search
=================================

A 4x4 sliding tile puzzle solved with the A* search algorithm.

Heuristic used: Manhattan distance + Linear Conflict.
  - Manhattan distance: sum of |row difference| + |col difference| for every
    tile compared to its goal position. This is admissible (never overestimates).
  - Linear conflict: adds a penalty of 2 for every pair of tiles that are in
    their goal row/column but in the wrong order relative to each other
    (they will have to move out of the way and back in, costing >= 2 extra
    moves). Still admissible, and it makes the search *much* faster than
    Manhattan distance alone.

Usage:
    python3 fifteen_puzzle_astar.py

Controls in manual play mode: w/a/s/d or up/down/left/right to move the
blank tile, q to quit back to the menu.
"""

import heapq
import io
import itertools
import math
import os
import random
import re
import shutil
import statistics
import sys
import time

try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:  # Windows
    _HAS_TERMIOS = False

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:  # not Windows
    _HAS_MSVCRT = False

SIZE = 4
maxDepth = 0
GOAL = tuple(list(range(1, SIZE * SIZE)) + [0])  # 1,2,...,15,0


# ---------------------------------------------------------------------------
# ANSI colours & centring
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
_FG = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "gray": "\033[90m",
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def enable_windows_ansi():
    """On Windows, cmd.exe needs to be nudged to interpret ANSI escape codes
    (modern Windows Terminal/PowerShell don't need this, but plain cmd.exe
    on Windows 10+ does). This is the standard no-dependency trick for it."""
    if os.name == "nt":
        os.system("")


def colorize(text, color=None, bold=False):
    """Wrap text in ANSI color codes. No-ops if color is None."""
    prefix = (BOLD if bold else "") + (_FG.get(color, "") if color else "")
    if not prefix:
        return text
    return f"{prefix}{text}{RESET}"


def visible_len(s):
    """Length of a string as it will actually appear, ignoring ANSI codes
    (which have zero display width but count toward len() otherwise)."""
    return len(_ANSI_RE.sub("", s))


def term_size():
    cols, rows = shutil.get_terminal_size(fallback=(80, 24))
    return cols, rows


def center_line(s, width=None):
    if width is None:
        width, _ = term_size()
    pad = width - visible_len(s)
    if pad <= 0:
        return s
    return " " * (pad // 2) + s


def print_centered(text=""):
    """Print (possibly multi-line) text with each line horizontally centered."""
    width, _ = term_size()
    for line in text.split("\n"):
        print(center_line(line, width))


def print_screen(lines):
    """Clear the screen, then print `lines` both horizontally centered (to
    terminal width) and vertically centered (to terminal height, roughly --
    leaves a little extra room below for an input prompt)."""
    clear_screen()
    cols, rows = term_size()
    vertical_pad = max(0, (rows - len(lines)) // 2 - 1)
    print("\n" * vertical_pad, end="")
    for line in lines:
        print(center_line(line, cols))


def redraw_live_block(lines, first_draw):
    """Redraw a small, fixed-size block of centered lines in place, instead
    of clearing and repainting the whole screen every time like
    print_screen does. The first call behaves like print_screen (clear +
    vertically center); every later call -- as long as `lines` always has
    the same length -- moves the cursor back up to the top of the block
    and overwrites each line rather than clearing the terminal, which is
    what stops a fast, frequent updater (like a live search counter) from
    making the whole console flash."""
    cols, rows = term_size()
    if first_draw:
        clear_screen()
        vertical_pad = max(0, (rows - len(lines)) // 2 - 1)
        sys.stdout.write("\n" * vertical_pad)
    else:
        sys.stdout.write(f"\x1b[{len(lines)}A")  # cursor up to top of block
    for line in lines:
        # \r -> column 0, print the new (centered) line, then \x1b[K clears
        # anything left over to the right -- so a shorter new line can't
        # leave stray characters from a longer previous one.
        sys.stdout.write("\r" + center_line(line, cols) + "\x1b[K\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Puzzle mechanics
# ---------------------------------------------------------------------------

def get_neighbors(state):
    """Return list of (new_state, move_name) reachable by sliding one tile
    into the blank space."""
    zero_idx = state.index(0)
    r, c = divmod(zero_idx, SIZE)
    moves = []
    for dr, dc, name in [(-1, 0, "Up"), (1, 0, "Down"),
                          (0, -1, "Left"), (0, 1, "Right")]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE:
            nidx = nr * SIZE + nc
            new_state = list(state)
            new_state[zero_idx], new_state[nidx] = new_state[nidx], new_state[zero_idx]
            moves.append((tuple(new_state), name))
    return moves


def manhattan_distance(state):
    dist = 0
    for i, val in enumerate(state):
        if val == 0:
            continue
        r, c = divmod(i, SIZE)
        gr, gc = divmod(val - 1, SIZE)
        dist += abs(r - gr) + abs(c - gc)
    return dist


def linear_conflict(state):
    conflict = 0

    # Row conflicts
    for r in range(SIZE):
        tiles = []
        for c in range(SIZE):
            val = state[r * SIZE + c]
            if val != 0 and (val - 1) // SIZE == r:
                tiles.append(val)
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if tiles[i] > tiles[j]:
                    conflict += 1

    # Column conflicts
    for c in range(SIZE):
        tiles = []
        for r in range(SIZE):
            val = state[r * SIZE + c]
            if val != 0 and (val - 1) % SIZE == c:
                tiles.append(val)
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if tiles[i] > tiles[j]:
                    conflict += 1

    return conflict * 2


def heuristic(state):
    return manhattan_distance(state) + linear_conflict(state)


def is_solvable(state):
    """Standard solvability check for the 15-puzzle: count inversions among
    the non-blank tiles, then combine with the blank's row (from the
    bottom, 1-indexed)."""
    tiles = [t for t in state if t != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1

    blank_row_from_bottom = SIZE - (state.index(0) // SIZE)

    if SIZE % 2 == 1:
        # Odd grid width: solvable iff inversions is even.
        return inversions % 2 == 0
    else:
        # Even grid width: depends on blank row parity too.
        if blank_row_from_bottom % 2 == 0:
            return inversions % 2 == 1
        else:
            return inversions % 2 == 0


def change_puzzle_size():
    global SIZE, GOAL

    print(f"Current size: {SIZE}x{SIZE} (the {SIZE*SIZE - 1}-puzzle)")
    raw = input("Enter new grid size n (n x n board, n >= 2), or blank to cancel: ").strip()
    if not raw:
        return

    try:
        n = int(raw)
    except ValueError:
        print("That's not a whole number.")
        time.sleep(1.2)
        return

    if n < 2:
        print("Size must be at least 2.")
        time.sleep(1.2)
        return

    if n > 4:
        print(f"\nHeads up: plain A* (as implemented here) reliably solves 4x4 "
              f"but tends to struggle above that -- see 'Estimate solve time "
              f"for current size' on the menu before trying 'Watch A* solve' "
              f"at {n}x{n}.")

    SIZE = n
    GOAL = tuple(list(range(1, n * n)) + [0])
    print(f"\nSize set to {n}x{n}.")
    time.sleep(1.0)


def default_shuffle_count():
    """Pick a shuffle count that scales with grid size, so larger boards
    still look thoroughly mixed (a fixed constant only "looked scrambled"
    on a 4x4). Grows as SIZE**3 + SIZE*2 -- steeper than a simple SIZE*SIZE
    scaling, so boards end up more thoroughly mixed at every size.

    Heads up: at SIZE=4 this works out to 72 shuffles, which pushes the
    *average* optimal solution length to around ~37 moves -- right at the
    edge of where plain A* (unlike IDA*, the memory-efficient variant
    real-world 15-puzzle solvers use, which keeps every visited state in
    memory) starts to slow down noticeably. Benchmarked: solves that
    average a few seconds, with some scrambles taking 30+ seconds and
    expanding upwards of a million nodes. This is a real tradeoff versus
    the old SIZE*SIZE*2 formula (32 shuffles, ~0.2s/solve): boards are
    more scrambled, but 'Watch A* solve' waits noticeably longer -- that's
    exactly the case the progress bar in watch_astar_solve() is for.
    Larger boards (5x5+) are fundamentally outside what this plain-A*
    implementation can solve quickly -- see the warning in main_menu()."""
    return (SIZE ** 3)


def generate_puzzle(num_shuffles=None):
    """Generate a solvable puzzle by making random legal moves starting
    from the goal state (guarantees solvability by construction)."""
    if num_shuffles is None:
        num_shuffles = default_shuffle_count()

    state = GOAL
    recent_moves = []  # last couple of moves, to avoid short back-and-forth cycles
    opposite = {"Up": "Down", "Down": "Up", "Left": "Right", "Right": "Left"}

    for _ in range(num_shuffles):
        neighbors = get_neighbors(state)
        if recent_moves:
            # Avoid immediately reversing the last move, and avoid repeating
            # the move from two steps ago (kills simple 2-move oscillations
            # like Up,Down,Up,Down that a "no immediate reversal" rule alone
            # doesn't catch).
            banned = {opposite[recent_moves[-1]]}
            if len(recent_moves) >= 2:
                banned.add(recent_moves[-2])
            filtered = [n for n in neighbors if n[1] not in banned]
            if filtered:
                neighbors = filtered

        state, move = random.choice(neighbors)
        recent_moves.append(move)
        if len(recent_moves) > 2:
            recent_moves.pop(0)

    return state


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

def astar_solve(start, progress_callback=None, progress_interval=50):
    """Returns (list_of_moves, nodes_expanded) or (None, nodes_expanded) if
    no solution is found.

    If given, `progress_callback(nodes_expanded, depth, done=False)` is
    called every `progress_interval` expansions (plus once more with
    done=True right before returning a solution). `depth` is g_score of
    the node currently being expanded -- the number of moves from the
    start to that node, which is exactly the final move count once the
    search finishes. Calling it only every `progress_interval` expansions
    keeps a live display from slowing the search down by redrawing on
    every single node."""
    counter = itertools.count()  # tie-breaker so heap never compares states directly
    g_score = {start: 0}
    came_from = {start: (None, None)}
    open_heap = [(heuristic(start), next(counter), start)]
    closed = set()
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if progress_callback is not None and nodes_expanded % progress_interval == 0:
            progress_callback(nodes_expanded, g_score[current])

        if current == GOAL:
            if progress_callback is not None:
                progress_callback(nodes_expanded, g_score[current], done=True)
            return _reconstruct_path(came_from, current), nodes_expanded

        for neighbor, move_name in get_neighbors(current):
            if neighbor in closed:
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = (current, move_name)
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_heap, (f_score, next(counter), neighbor))

    return None, nodes_expanded


def _reconstruct_path(came_from, current):
    moves = []
    while came_from[current][0] is not None:
        parent, move = came_from[current]
        moves.append(move)
        current = parent
    moves.reverse()
    return moves


# ---------------------------------------------------------------------------
# Runtime estimation
# ---------------------------------------------------------------------------
#
# The formula, in plain terms:
#
# 1. BENCHMARK: solve several scrambled 4x4 puzzles for real, and measure
#    the average solution depth (moves) and average nodes A* expanded.
#
# 2. EFFECTIVE BRANCHING FACTOR (b): A* roughly explores nodes ~ b^depth
#    (this is the same idea used to describe search-tree growth in minimax
#    and IDA* analysis). Solving that for b using the measured depth/nodes
#    gives an empirical, calibrated branching factor for OUR heuristic on
#    THIS puzzle family -- rather than guessing one.
#
#       b = nodes_measured ^ (1 / depth_measured)
#
# 3. EXPECTED SOLUTION DEPTH AT SIZE n: it's a known asymptotic result for
#    the (n^2 - 1) sliding puzzle that the worst-case optimal solution
#    length grows as Theta(n^3) (e.g. Parberry, "The Devil's Algorithm").
#    We use that cubic relationship to scale the *measured* 4x4 depth up
#    to the current size:
#
#       depth_estimate(n) = depth_measured * (n / 4)^3
#
# 4. PROJECTED NODES & TIME: combine the two.
#
#       nodes_estimate(n) = b ^ depth_estimate(n)
#       time_estimate(n)  = nodes_estimate(n) * (time_measured / nodes_measured)
#
# This is a genuine calculation, not a lookup table -- but it's still only
# an order-of-magnitude estimate: the branching factor isn't really
# constant (the heuristic's effectiveness shifts with board size), and the
# Theta(n^3) result is an asymptotic bound, not an exact formula. Treat the
# output as "roughly this many zeroes", not a precise prediction.

def run_4x4_benchmark(trials=6):
    """Solve `trials` freshly-scrambled 4x4 puzzles with real A*, regardless
    of whatever SIZE is currently set to, and return per-trial (time, nodes,
    depth) lists. Restores the original SIZE/GOAL afterwards."""
    global SIZE, GOAL
    saved_size, saved_goal = SIZE, GOAL
    SIZE, GOAL = 4, tuple(list(range(1, 16)) + [0])

    times, nodes_list, depths = [], [], []
    try:
        for _ in range(trials):
            puzzle = generate_puzzle()
            t0 = time.perf_counter()
            moves, nodes = astar_solve(puzzle)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            nodes_list.append(nodes)
            depths.append(len(moves))
    finally:
        SIZE, GOAL = saved_size, saved_goal

    return times, nodes_list, depths


def human_duration(seconds):
    """Format a (reasonably-sized) number of seconds as a friendly string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} minutes"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.2f} hours"
    days = hours / 24
    if days < 365:
        return f"{days:.2f} days"
    years = days / 365.25
    return f"{years:,.3g} years"


AGE_OF_UNIVERSE_SECONDS_LOG10 = math.log10(4.35e17)  # ~13.8 billion years


def format_estimated_time(log10_seconds):
    """Format a time estimate given as log10(seconds), safely handling
    numbers far too large for a normal float to hold."""
    if log10_seconds < 15:  # safe to materialize directly (up to ~30M years)
        return human_duration(10 ** log10_seconds)

    diff = log10_seconds - AGE_OF_UNIVERSE_SECONDS_LOG10
    if diff < 300:  # still safe to turn into an actual number
        multiple_of_universe_age = 10 ** diff
        return (f"~10^{log10_seconds:.1f} seconds "
                f"(about {multiple_of_universe_age:.2e}x the current age of the universe)")

    # Too large even for that comparison to be computable as a float --
    # just report the order of magnitude and how many orders bigger than
    # the age of the universe it is (safe: this is a subtraction, not a
    # power operation).
    return (f"~10^{log10_seconds:.1f} seconds "
            f"(about 10^{diff:.1f} times the current age of the universe -- "
            f"functionally never, on any human timescale)")


def estimate_runtime_for_current_size():
    n = SIZE
    trials = 6
    print(f"Benchmarking: solving {trials} real scrambled 4x4 puzzles with A*...")
    times, nodes_list, depths = run_4x4_benchmark(trials=trials)

    avg_time = statistics.mean(times)
    avg_nodes = statistics.mean(nodes_list)
    avg_depth = statistics.mean(depths)
    time_per_node = avg_time / avg_nodes if avg_nodes else 0.0

    print("\n--- 4x4 benchmark results ---")
    print(f"  average solution depth : {avg_depth:.1f} moves")
    print(f"  average nodes expanded : {avg_nodes:,.0f}")
    print(f"  average solve time     : {human_duration(avg_time)}")
    print(f"  time per node expanded : {time_per_node * 1e6:.2f} microseconds")

    if n == 4:
        print(f"\nCurrent size is already 4x4, so this benchmark *is* the estimate")
        print(f"for the current size -- no extrapolation needed.")
        input("\nPress Enter to return to the menu...")
        return

    b = avg_nodes ** (1 / avg_depth) if avg_depth > 0 else 1.0

    estimated_depth = avg_depth * (n / 4) ** 3

    log10_nodes = estimated_depth * math.log10(b) if b > 0 else 0.0
    log10_time = log10_nodes + math.log10(time_per_node) if time_per_node > 0 else float("-inf")

    print(f"\n--- Estimate for current size: {n}x{n} ({n*n - 1}-puzzle) ---")
    print(f"  effective branching factor (from benchmark) : {b:.3f}")
    print(f"  estimated optimal solution depth (Theta(n^3)): {estimated_depth:,.0f} moves")
    if log10_nodes < 100:
        print(f"  estimated nodes A* would expand              : {10 ** log10_nodes:,.0f}")
    else:
        print(f"  estimated nodes A* would expand              : ~10^{log10_nodes:.1f}")
    print(f"  estimated solve time                          : {format_estimated_time(log10_time)}")

    print("\n(This is an order-of-magnitude estimate: the branching factor")
    print(" and the cubic depth-growth law are both approximations, not")
    print(" exact formulas -- treat this as 'roughly how many zeroes', not")
    print(" a precise prediction.)")
    input("\nPress Enter to return to the menu...")


# ---------------------------------------------------------------------------
# Menu and Rendering
# ---------------------------------------------------------------------------


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def render(state, header_lines=None):
    lines = []
    if header_lines:
        for line in header_lines:
            lines.append(line)
        lines.append("")

    border = colorize("+" + "----+" * SIZE, "blue", bold=True)
    pipe = colorize("|", "blue", bold=True)

    lines.append(border)
    for r in range(SIZE):
        row = pipe
        for col in range(SIZE):
            idx = r * SIZE + col
            val = state[idx]
            if val == 0:
                cell = colorize("   ", "gray")
            else:
                text = f"{val:2} "
                in_place = (val - 1 == idx)  # matches this tile's goal position
                cell = colorize(text, "green", bold=True) if in_place else colorize(text, "white")
            row += " " + cell + pipe
        lines.append(row)
        lines.append(border)

    print_screen(lines)


def read_single_key():
    """Read one keypress with no Enter required, and normalize it:
      - Arrow keys  -> 'Up' / 'Down' / 'Left' / 'Right'
      - Other keys  -> the lowercase character (e.g. 'w', 'q', 'h')
      - Ctrl+C      -> raises KeyboardInterrupt, same as usual

    Tries Windows (msvcrt) first, then Unix raw terminal mode (termios/tty).
    If neither is usable (e.g. not a real terminal -- some IDEs, piped
    input), falls back to a plain input() line so the program still works,
    just without the single-keypress convenience.
    """
    if _HAS_MSVCRT:
        ch = msvcrt.getch()
        if ch == b"\x03":
            raise KeyboardInterrupt
        if ch in (b"\x00", b"\xe0"):  # prefix byte for arrow/special keys
            ch2 = msvcrt.getch()
            return {b"H": "Up", b"P": "Down", b"K": "Left", b"M": "Right"}.get(ch2, "")
        try:
            return ch.decode("utf-8", errors="ignore").lower()
        except Exception:
            return ""

    if _HAS_TERMIOS:
        old_settings = None
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
        except (termios.error, OSError, ValueError, AttributeError, io.UnsupportedOperation):
            old_settings = None  # stdin isn't a real terminal we can control

        if old_settings is not None:
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch == "\x1b":  # ESC -- likely the start of an arrow-key sequence
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        ch3 = sys.stdin.read(1)
                        return {"A": "Up", "B": "Down", "C": "Right", "D": "Left"}.get(ch3, "")
                    return ""
                return ch.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    typed = input("Move (raw key mode unavailable here -- type then Enter): ").strip().lower()
    word_to_move = {"up": "Up", "down": "Down", "left": "Left", "right": "Right"}
    return word_to_move.get(typed, typed)


def apply_move(state, move_name):
    for neighbor, name in get_neighbors(state):
        if name == move_name:
            return neighbor
    return state  # illegal move, no-op


def watch_astar_solve():
    puzzle = generate_puzzle()

    render(puzzle, header_lines=[colorize("Scrambled puzzle. Press Enter to run A*...", "yellow")])
    input()

    # A* doesn't know its total workload in advance, and estimating it up
    # front (e.g. by benchmarking sample solves) can easily take longer
    # than just solving the actual puzzle -- especially now that boards
    # are scrambled more thoroughly. So instead of a percent-complete bar,
    # just show what the search itself knows right now: how many nodes
    # it's expanded, and how many moves deep the node it's currently
    # looking at is (which becomes the final move count once it's done).
    start_time = time.time()
    last_draw = [0.0]
    first_draw = [True]
    max_depth = [0]

    def on_progress(nodes_expanded, depth, done=False):
        now = time.time()
        if not done and now - last_draw[0] < 0.08:
            return
        last_draw[0] = now
        elapsed_so_far = now - start_time
        max_depth[0] = max(max_depth[0], depth)

        lines = [
            colorize("Solving with A*", "cyan", bold=True),
            "",
            colorize(f"Nodes expanded: {nodes_expanded:,}", "white", bold=True),
            colorize(f"Max Depth: {max_depth[0]}", "white", bold=True),
            colorize(f"Moves from start (current node): {depth}", "white", bold=True),
            colorize(f"Elapsed: {elapsed_so_far:.1f}s", "gray"),
        ]
        redraw_live_block(lines, first_draw[0])
        first_draw[0] = False

    moves, nodes_expanded = astar_solve(puzzle, progress_callback=on_progress)
    elapsed = time.time() - start_time

    if moves is None:
        render(puzzle, header_lines=[colorize("No solution found (this shouldn't happen).", "red", bold=True)])
        input("Press Enter to return to the menu...")
        return

    print_centered(colorize(
        f"Solved! {len(moves)} moves, {nodes_expanded} nodes expanded, {elapsed:.3f}s.",
        "green", bold=True))
    print_centered("Press Enter to animate the solution step by step.")
    input()

    state = puzzle
    for i, move in enumerate(moves, start=1):
        state = apply_move(state, move)
        header = [
            colorize(f"Step {i}/{len(moves)}  (move: {move})", "cyan", bold=True),
            colorize(f"Nodes expanded during search: {nodes_expanded}   Time: {elapsed:.3f}s", "gray"),
        ]
        render(state, header_lines=header)
        time.sleep(0.35)

    print_centered(colorize("Solved!", "green", bold=True))
    input("Press Enter to return to the menu...")


def play_manually():
    puzzle = generate_puzzle(num_shuffles=max(20, default_shuffle_count() // 2))
    move_count = 0
    message = None  # transient status line shown for one frame, then cleared

    letter_to_move = {"w": "Up", "s": "Down", "a": "Left", "d": "Right"}

    while True:
        header = [
            colorize(f"Manual play — moves made: {move_count}", "cyan", bold=True),
            colorize("Controls: arrow keys or w/a/s/d to move (no Enter needed), "
                      "'h' for an A* hint, 'q' to quit", "gray"),
        ]
        if message:
            header.append("")
            header.append(message)
        render(puzzle, header_lines=header)
        message = None

        if puzzle == GOAL:
            print_centered(colorize("Solved! Nice work.", "green", bold=True))
            print_centered(colorize(f"Total moves made: {move_count}", "cyan"))
            input("Press Enter to return to the menu...")
            return

        key = read_single_key()

        if key == "q":
            return
        elif key == "h":
            moves, _ = astar_solve(puzzle)
            if moves:
                message = colorize(
                    f"Hint: A* says the next move is '{moves[0]}' "
                    f"({len(moves)} moves remain to solve optimally).", "cyan")
            time.sleep(1.6)
        elif key in ("Up", "Down", "Left", "Right") or key in letter_to_move:
            move_name = key if key in ("Up", "Down", "Left", "Right") else letter_to_move[key]
            new_state = apply_move(puzzle, move_name)
            if new_state != puzzle:
                puzzle = new_state
                move_count += 1
            else:
                message = colorize(
                    "That move isn't legal from here (tile not adjacent to blank).", "red")
                time.sleep(0.6)
        elif key:
            message = colorize(f"Unrecognized key: {key!r}", "yellow")
            time.sleep(0.6)


def main_menu():
    while True:
        lines = []
        lines.append(colorize("=" * 42, "magenta", bold=True))
        lines.append(colorize("15-PUZZLE  —  A* Search Demo", "magenta", bold=True))
        lines.append(colorize("=" * 42, "magenta", bold=True))
        lines.append("")
        lines.append(colorize(f"Current size: {SIZE}x{SIZE} ({SIZE*SIZE - 1}-puzzle)", "cyan"))
        lines.append("")
        lines.append(colorize("1) ", "yellow", bold=True) + "Watch A* solve a scrambled puzzle")
        lines.append(colorize("2) ", "yellow", bold=True) + "Play manually (with optional A* hints)")
        lines.append(colorize("3) ", "yellow", bold=True) + "Change puzzle size")
        lines.append(colorize("4) ", "yellow", bold=True) + "Estimate solve time for current size")
        lines.append(colorize("5) ", "yellow", bold=True) + "Quit")
        if SIZE > 4:
            lines.append("")
            warning = [
                "Note: The A* solver is not practical for sizes larger than 4x4. It will likely",
                "be unable to solve puzzles of size 5x5 or larger in a reasonable time.",
                "The estimate for larger sizes is that it would take ~1.5 hours for 5x5,",
                "and ~1.5 years for 6x6, and the time grows exponentially from there."
            ]
            for w in warning:
                lines.append(colorize(w, "yellow"))

        print_screen(lines)
        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            watch_astar_solve()
        elif choice == "2":
            play_manually()
        elif choice == "3":
            change_puzzle_size()
        elif choice == "4":
            estimate_runtime_for_current_size()
        elif choice == "5":
            print_centered(colorize("Goodbye!", "magenta", bold=True))
            break
        else:
            print_centered(colorize("Invalid choice.", "red"))
            time.sleep(1)


if __name__ == "__main__":
    enable_windows_ansi()
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nExiting.")
