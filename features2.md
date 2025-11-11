Status-related features (P1 and P2)
1. p1_major_status_infliction_rate: Fraction of P1’s attempts with major-status moves (Sleep/Freeze) that successfully inflict them on P2.
2. p1_cumulative_major_status_turns_pct: Percentage of battle turns where P1 suffers Sleep/Freeze.
3. p2_major_status_infliction_rate: Fraction of P2’s major-status attempts that succeed on P1.
4. p2_cumulative_major_status_turns_pct: Percentage of battle turns where P2 suffers Sleep/Freeze.
5. p1_bad_status_advantage: Difference in negative status exposure between P1 and P2.
6. p1_status_change: Number of times P1’s Pokémon status changes during the battle.
7. p2_status_change: Number of times P2’s Pokémon status changes.
8. status_change_diff: Difference in number of status changes between P1 and P2.
9. net_major_status_infliction: Net advantage in inflicting major statuses (P1 rate minus P2 rate).
10. net_major_status_suffering: Net difference in suffering major statuses (P2 pct minus P1 pct).
Expected damage and type advantage features
11. expected_damage_ratio_turn_1: Log ratio of expected damage P1 deals vs. P2 on turn 1 using stats, move power, and type multipliers.
12. p1_type_resistance: Reciprocal of average weaknesses in P1 team; higher means more resistant on average.
13. p1_type_weakness: Normalized average number of type weaknesses in P1 team.
14. p1_type_diversity: Count of unique Pokémon types present in P1 team.
15. p1_mean_stab: Total number of STAB moves used by P1.
16. p2_mean_stab: Total number of STAB moves used by P2.
17. diff_mean_stab: Difference in STAB counts (P1 minus P2).
18. p1_type_advantage: Mean type effectiveness of P1 attacks against P2’s types across turns.
19. p2_type_advantage: Mean type effectiveness of P2 attacks against P1’s types.
20. diff_type_advantage: Difference in type advantage metrics (P1 minus P2).
Team and lead statistics (static)
21. p1_max_offensive_stat: Highest Attack or Special Attack among P1’s team.
22. p1_max_speed_stat: Highest Speed stat among P1’s team.
23. p1_mean_hp: Average base HP of P1’s team.
24. p1_mean_spe: Average base Speed of P1’s team.
25. p1_mean_atk: Average base Attack of P1’s team.
26. p1_mean_def: Average base Defense of P1’s team.
27. p1_mean_sp: Average base Special (Special Defense) of P1’s team.
P1 vs P2 lead deltas
28. diff_hp: HP difference between the lead Pokémon (P1 minus P2).
29. diff_spe: Speed difference between lead Pokémon.
30. diff_atk: Attack difference between leads.
31. diff_def: Defense difference between leads.
32. diff_spd: Special Defense difference between leads.
Timeline-based dynamic stat deltas
33. mean_base_atk_diff_timeline: Mean Attack difference of active Pokémon over turns.
34. std_base_atk_diff_timeline: Variability in Attack difference across turns.
35. mean_base_spa_diff_timeline: Mean Special Attack difference over turns.
36. std_base_spa_diff_timeline: Variability in Special Attack difference.
37. mean_base_spe_diff_timeline: Mean Speed difference between active Pokémon over time.
38. std_base_spe_diff_timeline: Variability in Speed difference over time.
Boost-related dynamic features
39. p1_net_boost_sum: Total net advantage in stat boosts accumulated over time (sum of all P1 boost stages minus P2).
40. p1_max_offense_boost_diff: Maximum offensive stat-stage advantage (Attack + Special Attack) achieved by P1 compared to P2.
41. p1_max_speed_boost_diff: Maximum Speed boost advantage achieved by P1 compared to P2.
Priority-related features
42. priority_diff: Difference in average move priority between P1 and P2.
43. priority_rate_advantage: Fraction of turns where P1 has strictly higher move priority than P2.
HP-based progress features
44. hp_diff_mean: Mean HP percentage difference (P1 minus P2) over all turns.
45. p1_hp_advantage_mean: Proportion of turns where P1’s HP% exceeds P2’s.
46. p1_hp_std: Standard deviation of P1’s HP% over the battle.
47. p2_hp_std: Standard deviation of P2’s HP%.
48. hp_delta_std: Standard deviation of HP difference (P1 minus P2) over time.
49. hp_delta_trend: Linear trend (slope) of HP difference over turns.
50. hp_advantage_trend: Linear slope of P1 HP advantage (P1 HP% - P2 HP%) over turns.
Battle outcome flow features
51. p1_n_pokemon_use: Number of unique Pokémon P1 sends into battle.
52. p2_n_pokemon_use: Number of unique Pokémon P2 uses.
53. diff_final_schieramento: Difference in number of deployed Pokémon.
54. nr_pokemon_sconfitti_p1: Number of P1 Pokémon knocked out.
55. nr_pokemon_sconfitti_p2: Number of P2 Pokémon knocked out.
56. nr_pokemon_sconfitti_diff: Difference in KO count (P1 minus P2).
57. p1_pct_final_hp: Final summed HP% remaining for P1’s team.
58. p2_pct_final_hp: Final summed HP% remaining for P2’s team.
59. diff_final_hp: Difference in final HP sums (P1 minus P2).
60. battle_duration: Number of turns during which both Pokémon are still alive.
61. hp_loss_rate: Final HP difference normalized by battle duration.
62. early_hp_mean_diff: Mean HP difference in early turns of the battle.
63. late_hp_mean_diff: Mean HP difference in late turns of the battle.
Speed-based features (team-level)
64. p1_avg_speed_stat_battaglia: Fraction of P1 team with Speed above a medium threshold.
65. p1_avg_high_speed_stat_battaglia: Fraction of P1 team with Speed above a high threshold.
Interaction features
66. p1_max_speed_offense_product: Product of P1’s highest Speed and highest offensive stat.
67. p1_final_hp_per_ko: Final HP total divided by (KO count + 1).
Labels / Metadata
68. battle_id: Unique identifier of the battle.
69. player_won: Target label (1 if P1 wins, 0 if P2 wins).