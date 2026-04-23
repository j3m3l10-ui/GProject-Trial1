# Safe Hardware Testing Guide — Tomato Harvesting Robot

## Overview
The `main.py` now includes **multiple safety modes** to allow you to test the robotic arm safely on real hardware **without risk of unintended movements or cuts**.

## Safety Flags

### 1. **--dry-run** (Recommended for first-time testing)
**What it does:** Logs all planned movements but **doesn't send any servo commands**. Perfect for verifying the logic flow without hardware motion.

```bash
python main.py --dry-run
```

**Output:**
- Vision detection works normally (camera runs)
- Arm inverse kinematics solved normally (computed but not executed)
- All servo commands logged with `[DRY RUN]` prefix
- No actual arm movement
- Can watch live detection on screen

**Typical flow:**
1. Scan phase runs for 3 seconds (live camera)
2. Confirm phase runs for 3 seconds (live camera)
3. Harvest sequence simulated — shows what would happen
4. Safe to test on any setup

---

### 2. **--no-cut**
**What it does:** Arm moves to target and positions for cutting, **but gripper never closes**. Tomato stays on plant.

```bash
python main.py --no-cut
```

**Output:**
- Full arm movement (real hardware)
- Gripper **opens** for approach
- Gripper **never closes** (skip cutting)
- Gripper opens at end (no-op)
- No tomato damage

**Typical flow:**
1. Full harvest cycle
2. Arm moves to each tomato
3. At moment of cut: `[SKIPPED-DRY-RUN/NO-CUT] CUTTING stem...`
4. Arm retracts to home
5. Tomatoes remain intact

---

### 3. **--single-pass**
**What it does:** Run **exactly 1 harvest cycle** then exit (no infinite loop).

```bash
python main.py --single-pass
```

**Typical flow:**
1. Scan phase (3s)
2. Confirm phase (3s)
3. Harvest up to 3 tomatoes
4. Return home
5. **Exit immediately**

---

### 4. **--test-cycles N**
**What it does:** Limit total execution to N harvest cycles.

```bash
python main.py --test-cycles 2
```

**Runs:** Cycle 1 → Cycle 2 → Exit

---

### 5. **--no-confirm**
**What it does:** Auto-confirm all prompts (skip manual yes/no questions at cut time).

```bash
python main.py --no-confirm
```

**By default:** Before each cut, user is asked: `READY TO CUT STEM for tomato #1. Proceed? [y/N]:`

**With --no-confirm:** Automatically proceeds (assumes yes)

---

## Recommended Testing Sequence

### **Stage 1: Logic Verification (No Hardware Risk)**
```bash
python main.py --dry-run --single-pass
```
✓ See the full workflow without any servo movement  
✓ Verify detection, IK solving, depth calculation  
✓ Expected: Logs show all steps, no hardware moves

---

### **Stage 2: Movement Verification (Real Hardware, No Cutting)**
```bash
python main.py --no-cut --single-pass
```
✓ Arm actually moves to detected tomatoes  
✓ Gripper positions for cut but never closes  
✓ Full orchestration tested in real hardware  
✓ **Tomatoes stay on plant (safe)**  
✓ Expected: Arm moves smoothly, returns home safely

---

### **Stage 3: Single Harvest Attempt (With Cutting)**
```bash
python main.py --single-pass
```
✓ Full harvest cycle with real cutting  
✓ Tests gripper close/open  
✓ **Must have at least 1 tomato in view**  
✓ Expected: 1 cycle complete, system parks safely

---

### **Stage 4: Limited Multi-Cycle Testing (With Cutting)**
```bash
python main.py --test-cycles 3
```
✓ Run 3 full harvest cycles  
✓ Tests repeated cycles, farm-level robustness  
✓ Expected: System cycles through scan/confirm/harvest 3 times, parks

---

### **Stage 5: Production Continuous Mode**
```bash
python main.py
```
✓ Infinite harvest loop  
✓ Runs until user presses **Q** in the video window or **Ctrl+C** in terminal  
✓ Can harvest for hours on a productive plant

---

## Combining Flags

You can **combine multiple flags** for precise control:

### Example 1: Dry run with limited cycles
```bash
python main.py --dry-run --test-cycles 3
```
Simulates 3 harvest cycles without touching hardware.

### Example 2: Real hardware test with 1 cycle
```bash
python main.py --no-cut --single-pass
```
Full arm movement, gripper prep, but no actual cutting — perfect for final pre-harvest checkout.

### Example 3: Full cut with automatic yes
```bash
python main.py --single-pass --no-confirm
```
One full harvest cycle, auto-proceed on all prompts.

---

## Output Examples

### Dry-Run Output (Fragment)
```
[2026-04-23 10:15:33] [INFO] Starting 3-tomato batch harvest system in REAL mode
████████████████████████████████████████████████████████
                 ⚠️  DRY RUN MODE ACTIVE ⚠️
  All movements will be LOGGED NOT EXECUTED on hardware
████████████████████████████████████████████████████████

[2026-04-23 10:15:33] [INFO] SINGLE PASS MODE — Will run 1 cycle then exit
[2026-04-23 10:15:34] [INFO] ============================================================
[2026-04-23 10:15:34] [INFO] CYCLE 1:  SCANNING for ripe tomatoes (3.0s window)...
[2026-04-23 10:15:34] [DRY RUN] Resetting arm to home base for scan
...
[2026-04-23 10:15:37] [INFO] Detected 1 stable tomato(es), nearest-first.
[2026-04-23 10:15:37] [INFO]   #1: distance=28.3cm, reachable=True, conf=0.92, sightings=8
[2026-04-23 10:15:40] [DRY RUN] [1/1] Moving to cut point
[2026-04-23 10:15:41] [DRY RUN] [1/1] Closing gripper (500ms)
[2026-04-23 10:15:42] [DRY RUN] [1/1] Opening gripper (400ms)
[2026-04-23 10:15:42] [INFO] [1/1] [SKIPPED-DRY-RUN/NO-CUT] stem...
```

### No-Cut Output (Fragment)
```
[2026-04-23 10:16:15] [INFO] Starting 3-tomato batch harvest system in REAL mode
NO-CUT MODE ACTIVE — Gripper will not cut tomatoes
[2026-04-23 10:16:15] [INFO] SINGLE PASS MODE — Will run 1 cycle then exit
...
[2026-04-23 10:16:25] [INFO] [1/1] Approaching stem...
[2026-04-23 10:16:26] [INFO] [1/1] [SKIPPED-DRY-RUN/NO-CUT] CUTTING stem...
[2026-04-23 10:16:26] [INFO] Cycle 1 complete. Ready for next scan.
[2026-04-23 10:16:26] [INFO] Single-pass mode: exiting after 1 cycle
```

---

## Common Scenarios

### Scenario A: "I want to watch the arm move but NOT cut tomatoes"
```bash
python main.py --no-cut --single-pass
```

### Scenario B: "I want to see the full logic flow on-screen before touching hardware"
```bash
python main.py --dry-run --single-pass
```

### Scenario C: "I want to test the orchestration with light load (1-3 cycles)"
```bash
python main.py --test-cycles 3
```

### Scenario D: "I'm confident. Go full production (harvest all day)."
```bash
python main.py
```

---

## Emergency Stop
At any time during execution:
- **Press Q** in the live camera window to stop gracefully
- **Press Ctrl+C** in the terminal to force emergency stop
- System will park all servos safely and exit

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Arm doesn't move (dry-run) | ✓ This is expected! Check logs for `[DRY RUN]` messages |
| Gripper doesn't cut (no-cut) | ✓ This is expected! Use without `--no-cut` to enable cutting |
| System asks for confirmation at every cut | Remove `--no-confirm` to get auto-yes, or just keep answering |
| No tomatoes detected | Place 1+ ripe tomatoes (~30cm away), ensure good lighting |
| Out-of-reach warning | Move plant/arm setup so tomatoes are ≤36cm from camera lens |

---

## Calibration Tip

After verifying movement with dry-run and no-cut, the **first real harvest** should be with `--single-pass` to ensure:
- ✓ Gripper closes properly (cuts stem)
- ✓ Tomato falls into collection net
- ✓ Arm retracts cleanly
- ✓ System parks safely

Then proceed to production mode: `python main.py`

---

**You're ready to harvest safely!** 🍅
