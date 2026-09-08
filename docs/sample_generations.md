# Sample generations

Base: `Qwen/Qwen2.5-0.5B` vs the same model + LoRA adapter, on held-out test rows.

Most gains are convention-fixing: `'x'` → `"x"`, casing, and not over-selecting
columns (samples 1, 3, 5, 8–15). Samples 2, 13 and 14 show genuine schema-grounding
fixes. Samples **4 and 7 are still wrong after fine-tuning** (`a_` vs `kickoff_[a_]`;
`MAX` vs `AVG`) — kept here rather than cherry-picked away.


## 1. Who was the celebrity who exited on day 13?

**Schema:** `CREATE TABLE table_name_35 (celebrity VARCHAR, exited VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT celebrity FROM table_name_35 WHERE exited = "day 13"` |
| base | `SELECT celebrity FROM table_name_35 WHERE exited = '13'` |
| finetuned | `SELECT celebrity FROM table_name_35 WHERE exited = "day 13"` |

## 2. who is the opponent when the method is tko (punches) at 4:26 of round 1?

**Schema:** `CREATE TABLE table_name_82 (opponent VARCHAR, method VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT opponent FROM table_name_82 WHERE method = "tko (punches) at 4:26 of round 1"` |
| base | `SELECT opponent FROM table_name_82 WHERE method = 'tko' AND round = 4 AND time = 26` |
| finetuned | `SELECT opponent FROM table_name_82 WHERE method = "tko (punches) at 4:26 of round 1"` |

## 3. Which years did the Tampa Bay Storm win the Championships?

**Schema:** `CREATE TABLE table_name_71 (championships__years_ VARCHAR, club VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT championships__years_ FROM table_name_71 WHERE club = "tampa bay storm"` |
| base | `SELECT championship_years__years_ FROM table_name_71 WHERE club = 'Tampa Bay Storm'` |
| finetuned | `SELECT championships__years_ FROM table_name_71 WHERE club = "tampa bay storm"` |

## 4. What date had a kickoff of 1:00 and a result of l 27-14?

**Schema:** `CREATE TABLE table_name_81 (date VARCHAR, result VARCHAR, kickoff_ VARCHAR, a_ VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT date FROM table_name_81 WHERE kickoff_[a_] = "1:00" AND result = "l 27-14"` |
| base | `SELECT date FROM table_name_81 WHERE result = 'l 27-14' AND kickoff_ = '1:00'` |
| finetuned | `SELECT date FROM table_name_81 WHERE kickoff_ = "1:00" AND a_ = "l 27-14"` |

## 5. What is Aberdeen team's date of vacancy?

**Schema:** `CREATE TABLE table_name_37 (date_of_vacancy VARCHAR, team VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT date_of_vacancy FROM table_name_37 WHERE team = "aberdeen"` |
| base | `SELECT date_of_vacancy FROM table_name_37 WHERE team = 'Aberdeen'` |
| finetuned | `SELECT date_of_vacancy FROM table_name_37 WHERE team = "aberdeen"` |

## 6. What is the production when engine is 2.7l, and acceleration 0–100km/h (0–62mph) is 8.5 s?

**Schema:** `CREATE TABLE table_name_5 (production VARCHAR, engine VARCHAR, acceleration_0_100km_h__0_62mph_ VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT production FROM table_name_5 WHERE engine = "2.7l" AND acceleration_0_100km_h__0_62mph_ = "8.5 s"` |
| base | `SELECT production FROM table_name_5 WHERE engine = 2.7l AND acceleration_0_100km_h__0_62mph_ = 8.5` |
| finetuned | `SELECT production FROM table_name_5 WHERE engine = "2.7l" AND acceleration_0_100km_h__0_62mph_ = "8.5 s"` |

## 7. Which Pos has a Dutch Cup of winner, and a Tier larger than 1?

**Schema:** `CREATE TABLE table_name_1 (pos INTEGER, dutch_cup VARCHAR, tier VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT AVG(pos) FROM table_name_1 WHERE dutch_cup = "winner" AND tier > 1` |
| base | `SELECT pos FROM table_name_1 WHERE dutch_cup = 'winner' AND tier > 1` |
| finetuned | `SELECT MAX(pos) FROM table_name_1 WHERE dutch_cup = "winner" AND tier > 1` |

## 8. Who had the placekicker position with a round above 10?

**Schema:** `CREATE TABLE table_name_64 (name VARCHAR, round VARCHAR, position VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT name FROM table_name_64 WHERE round > 10 AND position = "placekicker"` |
| base | `SELECT name, round FROM table_name_64 WHERE round > 10 AND position = 'placekicker'` |
| finetuned | `SELECT name FROM table_name_64 WHERE round > 10 AND position = "placekicker"` |

## 9. What Date is Week 1?

**Schema:** `CREATE TABLE table_name_54 (date VARCHAR, week VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT date FROM table_name_54 WHERE week = 1` |
| base | `SELECT date FROM table_name_54 WHERE week = '1'` |
| finetuned | `SELECT date FROM table_name_54 WHERE week = 1` |

## 10. On what date did the New York Giants host a game?

**Schema:** `CREATE TABLE table_name_42 (date VARCHAR, host_team VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT date FROM table_name_42 WHERE host_team = "new york giants"` |
| base | `SELECT date FROM table_name_42 WHERE host_team = 'New York Giants'` |
| finetuned | `SELECT date FROM table_name_42 WHERE host_team = "new york giants"` |

## 11. Who is the co-presenter of the series Seven (2007)?

**Schema:** `CREATE TABLE table_14345690_15 (co_presenter VARCHAR, series VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT co_presenter FROM table_14345690_15 WHERE series = "Seven (2007)"` |
| base | `SELECT co_presenter FROM table_14345690_15 WHERE series = 'Seven (2007)'` |
| finetuned | `SELECT co_presenter FROM table_14345690_15 WHERE series = "Seven (2007)"` |

## 12. What is the team's record on may 12 when they play the giants?

**Schema:** `CREATE TABLE table_name_58 (record VARCHAR, opponent VARCHAR, date VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT record FROM table_name_58 WHERE opponent = "giants" AND date = "may 12"` |
| base | `SELECT record FROM table_name_58 WHERE opponent = 'giants' AND date = 'may 12'` |
| finetuned | `SELECT record FROM table_name_58 WHERE opponent = "giants" AND date = "may 12"` |

## 13. Which young classification has an aggressive rider of Tim Johnson?

**Schema:** `CREATE TABLE table_name_89 (young_classification VARCHAR, aggressive_rider VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT young_classification FROM table_name_89 WHERE aggressive_rider = "tim johnson"` |
| base | `SELECT young_classification, aggressive_rider FROM table_name_89 WHERE young_classification = 'young' AND aggressive_rider = 'Tim Johnson'` |
| finetuned | `SELECT young_classification FROM table_name_89 WHERE aggressive_rider = "tim johnson"` |

## 14. What years does Our Lady Sacred Heart School, which is state integrated, have?

**Schema:** `CREATE TABLE table_name_47 (years VARCHAR, authority VARCHAR, name VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT years FROM table_name_47 WHERE authority = "state integrated" AND name = "our lady sacred heart school"` |
| base | `SELECT years FROM table_name_47 WHERE authority = 'Our Lady Sacred Heart School, which is state integrated'` |
| finetuned | `SELECT years FROM table_name_47 WHERE authority = "state integrated" AND name = "our lady sacred heart school"` |

## 15. What was the surface on march 2, 1997?

**Schema:** `CREATE TABLE table_name_91 (surface VARCHAR, date VARCHAR)`


| | SQL |
|---|---|
| gold | `SELECT surface FROM table_name_91 WHERE date = "march 2, 1997"` |
| base | `SELECT surface FROM table_name_91 WHERE date = 'March 2, 1997'` |
| finetuned | `SELECT surface FROM table_name_91 WHERE date = "march 2, 1997"` |