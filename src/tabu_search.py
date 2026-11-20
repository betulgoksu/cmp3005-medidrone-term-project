"""Route-based Tabu Search.

Planned:
- >= 3 moves (relocate, swap, 2-opt*), each with O(1) delta evaluation
- tabu list with tenure, aspiration criterion
- capacity handling (penalty or feasibility-only)
- run_tabu(instance, seed, budget) -> routes
"""
