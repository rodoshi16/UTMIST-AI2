# Custom Reward Function Analysis

## Issues Found

### 🚨 **Critical Issue: Duplicate Function Definitions**

All four reward functions you asked me to review exist **twice** in your code:

1. **Lines 459-494**: First definitions (wrong docstrings, looks like copy-paste errors)
2. **Lines 544-622**: Second definitions (correct docstrings)

The functions in the `gen_reward_manager()` at line 627 are using the SECOND definitions (lines 544-622), but the duplicates are confusing and could cause issues.

---

## Function-by-Function Analysis

### 1. `target_height_reward` (Lines 544-559)

**Current Implementation:**
```python
return -((obj.body.position.y - target_height)**2)
```

**Issues:**
- ✅ **This is well-implemented** - negative L2 squared distance is a standard reward formulation
- ✅ Returns a negative reward (penalty) that gets smaller as player approaches target
- ✅ Smooth and continuous, which is good for RL

**Usage Note:** 
- Currently used with weight=0.0 in line 629, so it's effectively disabled

---

### 2. `head_to_middle_reward` (Lines 561-580)

**Current Implementation:**
```python
multiplier = -1 if player.body.position.x > 0 else 1
reward = multiplier * (player.body.position.x - player.prev_x)
return reward
```

**Issues:**
- ⚠️ **Has a bug**: Calculates velocity in the wrong direction

**The Problem:**
- When `player.position.x > 0` (right side), multiplier is -1
- When `player.position.x < 0` (left side), multiplier is 1
- This means:
  - Player on RIGHT side: reward = `-1 * (current - prev)` = `prev - current`
  - Player on LEFT side: reward = `1 * (current - prev)` = `current - prev`

**What this actually does:**
- RIGHT side: Rewards moving LEFT (toward middle)
- LEFT side: Rewards moving RIGHT (toward middle)
- ✅ **This IS correct!** The logic is right

**However:**
- This only rewards horizontal movement toward x=0
- If player is already at middle (x ≈ 0), the multiplier flips sign based on tiny differences
- Creates discontinuity at x=0

**Potential Issues:**
- Discontinuity at x=0 could confuse the agent
- No bounds checking - what if arena is larger than expected?

**Better Implementation Suggestion:**
```python
def head_to_middle_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    
    # Distance from middle
    dist_from_middle = abs(player.body.position.x)
    
    # Reward for reducing distance (moving toward middle)
    prev_dist = abs(player.prev_x)
    reward = prev_dist - dist_from_middle
    
    return reward
```

**Why this is better:**
- Continuous everywhere
- Directly measures progress toward middle
- No sign flipping issues
- Works regardless of arena size

---

### 3. `head_to_opponent` (Lines 582-602)

**Current Implementation:**
```python
multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
reward = multiplier * (player.body.position.x - player.prev_x)
return reward
```

**Issues:**
- ⚠️ **Same discontinuity problem** as head_to_middle
- ⚠️ **Only considers horizontal movement**

**What this does:**
- If player is RIGHT of opponent, reward = `-1 * (current - prev)` = reward for moving left (toward opponent)
- If player is LEFT of opponent, reward = `1 * (current - prev)` = reward for moving right (toward opponent)
- ✅ Logic is correct for horizontal component

**Problems:**
1. **No vertical component** - doesn't reward moving up/down toward opponent
2. **Discontinuity** when player.x exactly equals opponent.x
3. **Not distance-based** - moving 1 unit when 100 units away gives same reward as moving 1 unit when 1 unit away

**Better Implementation Suggestion:**
```python
def head_to_opponent(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Current distance
    current_dist = np.sqrt(
        (player.body.position.x - opponent.body.position.x)**2 +
        (player.body.position.y - opponent.body.position.y)**2
    )
    
    # Previous distance
    prev_dist = np.sqrt(
        (player.prev_x - opponent.body.position.x)**2 +
        (player.prev_y - opponent.body.position.y)**2
    )
    
    # Reward for getting closer
    reward = prev_dist - current_dist
    
    return reward
```

**Why this is better:**
- Considers both horizontal AND vertical movement
- Continuous everywhere
- Scales with distance (closer = more reward for same movement)
- Natural interpretation: reward proportional to distance reduction

**OR, for movement-based (similar to current):**
```python
def head_to_opponent(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Direction vector toward opponent
    dx = opponent.body.position.x - player.body.position.x
    dy = opponent.body.position.y - player.body.position.y
    
    # Normalize (handle division by zero)
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.001:
        return 0  # Already on top of opponent
    
    dx_norm = dx / dist
    dy_norm = dy / dist
    
    # Movement this frame
    vel_x = player.body.position.x - player.prev_x
    vel_y = player.body.position.y - player.prev_y
    
    # Reward velocity in direction of opponent
    reward = dx_norm * vel_x + dy_norm * vel_y
    
    return reward
```

---

### 4. `taunt_reward` (Lines 604-622)

**Current Implementation:**
```python
reward = 1 if isinstance(player.state, TauntState) else 0.0
return reward * env.dt
```

**Issues:**
- ✅ **This is well-implemented**
- ✅ Returns a fixed reward for being in taunt state
- ✅ Multiplied by dt for proper time integration

**One potential improvement:**
If you want to discourage spamming taunt, you could add a small negative reward when entering taunt from certain states:

```python
# Could track previous state and penalize frequent taunting
# But current implementation is fine for basic use
```

---

## Summary Recommendations

| Function | Current Quality | Recommended Action |
|----------|----------------|-------------------|
| `target_height_reward` | ✅ Good | Keep as-is |
| `head_to_middle_reward` | ⚠️ Works but has issues | **Rewrite** to distance-based |
| `head_to_opponent` | ⚠️ Only 1D, has issues | **Rewrite** to 2D distance-based |
| `taunt_reward` | ✅ Good | Keep as-is |

## General Recommendations

1. **Delete duplicate functions** (lines 459-494) to avoid confusion
2. **Fix head_to_middle** to use distance-based formulation
3. **Fix head_to_opponent** to consider both X and Y movement
4. **Test reward scaling** - make sure magnitudes are appropriate relative to other rewards

## Context: How prev_x Works

From code analysis:
- `prev_x` is updated in `physics_process()` after physics step completes (line 3933)
- This means `position.x - prev_x` gives the **change in position this frame** (velocity * dt)
- This is correct for calculating movement rewards

## Note on Weights

Your current weights in `gen_reward_manager()`:
- `head_to_middle_reward`: 0.01
- `head_to_opponent`: 0.05
- `taunt_reward`: 0.2

These seem reasonable for encouraging engagement (head_to_opponent) without overwhelming other signals.

