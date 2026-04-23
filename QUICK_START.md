# Quick Start Cheat Sheet — Safe Hardware Testing

## 5-Step Safe Testing Path

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Verify Logic (No Hardware Movement)                │
│ $ python main.py --dry-run --single-pass                   │
│ ✓ See all planned moves in logs                            │
│ ✓ Watch live detection on camera                           │
│ ✓ Check arm IK solutions                                   │
│ Duration: ~10 seconds                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Test Real Movement (No Cutting)                    │
│ $ python main.py --no-cut --single-pass                    │
│ ✓ Arm moves to target positions                            │
│ ✓ Gripper approaches but NEVER closes                      │
│ ✓ Full orchestration on real hardware                      │
│ ✓ Tomatoes stay safe on plant                              │
│ Duration: ~1-2 minutes                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Dry-Run Full Cycle (3+ tomatoes simulated)         │
│ $ python main.py --dry-run --test-cycles 2                 │
│ ✓ See how multi-tomato harvest would execute               │
│ ✓ Ranking, out-of-reach checks                             │
│ ✓ Return-to-home sequence                                  │
│ Duration: ~30 seconds per cycle                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Real Single Harvest (With Cutting!)                │
│ $ python main.py --single-pass                             │
│ ✓ Full harvest with real gripper close                     │
│ ✓ Will prompt before each cut: answer Y or N               │
│ ✓ Tomato collection validated                              │
│ Duration: ~2-3 minutes                                     │
│ ⚠️  NEEDS: At least 1 ripe tomato in view                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Production Mode (Infinite Loops)                   │
│ $ python main.py                                           │
│ ✓ Full autonomous harvest                                  │
│ ✓ Continuous until manual stop (Q or Ctrl+C)               │
│ ✓ Press Q in video window or Ctrl+C in terminal            │
│ Perfect for: Productive plants (many tomatoes ready)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Flag Reference Card

| Flag | Effect | Example |
|------|--------|---------|
| `--dry-run` | Log moves, don't execute | `python main.py --dry-run` |
| `--no-cut` | Move arm, skip gripper close | `python main.py --no-cut` |
| `--single-pass` | Run 1 cycle then exit | `python main.py --single-pass` |
| `--test-cycles N` | Limit to N cycles | `python main.py --test-cycles 3` |
| `--no-confirm` | Auto-yes all prompts | `python main.py --no-confirm` |
| `--sim` | Simulation mode (no I2C) | `python main.py --sim` |
| `--camera N` | Use camera index N | `python main.py --camera 1` |

---

## Common Commands

### 🧪 Testing (Most Common)
```bash
# Preview without any hardware movement
python main.py --dry-run --single-pass

# Test movement without cutting
python main.py --no-cut --single-pass

# Real single harvest with prompts
python main.py --single-pass

# Real harvest auto-confirming cuts
python main.py --single-pass --no-confirm
```

### 📊 Multi-Cycle Testing
```bash
# Dry-run 3 cycles (log only)
python main.py --dry-run --test-cycles 3

# Real 3 cycles with cutting
python main.py --test-cycles 3

# Unlimited real cycles (Ctrl+C to stop)
python main.py
```

### 🔧 Edge Cases
```bash
# Combine dry-run + no-confirm (for scripting)
python main.py --dry-run --no-confirm --test-cycles 5

# Use camera 1 instead of 0
python main.py --camera 1 --single-pass

# Simulation mode (no real I2C/hardware needed)
python main.py --sim --dry-run
```

---

## Log Output Interpretation

### ✅ Success Indicators
```
[1/1] IK: solved=True, err=0.0024m, iters=12
[1/1] Approaching stem...
[1/1] CUTTING stem...
[1/1] Cut complete!
```

### ⚠️ Warning Signs
```
[WARN] [1/1] Target OUT OF REACH — skipping
→ Tomato too far from arm. Move closer or reposition.

[WARN] [1/1] IK error too large — skipping
→ Target position not reachable. Check arm limits.

[WARN] [1/1] Target is 42.5cm from lens (>36cm)
→ Must place plant within 36cm of camera.
```

### 🔴 Dry-Run Indicators
```
[DRY RUN] Trajectory: 25 steps to angles [1.23, 0.45, ...] in 800ms
[DRY RUN] Closing gripper (500ms)
```
→ These are *simulated*—no hardware moved.

---

## Pre-Flight Checklist

Before running STEP 4+ with real hardware:

- [ ] Camera mounted and focused on plant
- [ ] Arm powered on and calibrated
- [ ] At least 1 ripe tomato visible
- [ ] Collection net positioned below arm
- [ ] No obstacles in arm's workspace
- [ ] Emergency stop accessible (press Q or Ctrl+C anytime)
- [ ] Comfortable with arm movements
- [ ] Gripper tested manually (open/close commands work)

---

## Safety Reminders

1. **Dry-run is FREE**: Use it liberally before hardware testing
2. **No-cut is SAFE**: Perfect for verifying movement without damage  
3. **S single-pass is PREDICTABLE**: Known to run exactly 1 cycle
4. **Press Q ANYTIME**: Video window responsive to emergency stop
5. **Press Ctrl+C ANYTIME**: Terminal also accepts Ctrl+C for emergency

---

**Ready?**
```
python main.py --dry-run --single-pass
```

👉 Start here! No risk, full preview. 🍅
