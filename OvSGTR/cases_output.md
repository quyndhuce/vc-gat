### Case 1: Den Ngoc Son (Hanoi)
**Văn bản nguồn:**
> "Ngoc Son Temple is a temple on Jade Island in Hoan Kiem Lake, Hanoi. The temple is dedicated to Tran Hung Dao, a general who defeated the Mongols in the 13th century. It is connected to the shore by The Huc Bridge, a wooden bridge painted bright red that crosses the lake. The temple was built in 1841 in the traditional architectural style of the Nguyen dynasty. Turtle Tower, built in 1886, stands on a small islet in the southern part of the lake. The complex was recognized as a special national relic in 2013."

**Đồ thị tri thức trích xuất (21 facts, 17 nodes):**

| Head | Quan hệ | Tail | Score (Gốc) | $\mu_F$ (Fuzzy) | Trạng thái |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **The Huc Bridge** | `--[made_of]-->` | `wood` | 0.6893 | 0.5514 | `[Caption OK]` |
| **The Huc Bridge** | `--[crosses]-->` | `Hoan Kiem Lake` | 0.5859 | 0.4687 | `[Caption OK]` |
| **Ngoc Son Temple** | `--[dedicated_to]-->` | `Tran Hung Dao` | 0.5122 | 0.4098 | `[Caption OK]` |
| **Turtle Tower** | `--[built_in]-->` | `1886` | 0.5001 | 0.4001 | `[Caption OK]` |
| **Mongols** | `--[defeated_in]-->` | `13th century` | 0.4577 | 0.3662 | `[Caption OK]` |
| **Tran Hung Dao** | `--[defeated]-->` | `Mongols` | 0.3914 | 0.3131 | `[Caption OK]` |
| **The Huc Bridge** | `--[connects]-->` | `Ngoc Son Temple` | 0.3259 | 0.2607 | `[Caption OK]` |
| **Ngoc Son Temple** | `--[built_in]-->` | `1841` | 0.2746 | 0.2197 | `[Caption OK]` |
| **Ngoc Son Temple** | `--[situated_on]-->` | `Jade Island` | 0.2584 | 0.2067 | `[Caption OK]` |
| **Ngoc Son Temple** | `--[architectural_style]-->` | `traditional architectural style` | 0.2148 | 0.1718 | `[Caption OK]` |
| **Ngoc Son Temple** | `--[recognized_as]-->` | `special national relic` | 0.1820 | 0.1456 | `[Candidate]` |
| **Tran Hung Dao** | `--[defeated_in]-->` | `13th century` | 0.1699 | 0.1359 | `[Candidate]` |
| **Ngoc Son Temple** | `--[recognized_in]-->` | `2013` | 0.1414 | 0.1131 | `[Candidate]` |
| **Ngoc Son Temple** | `--[located_in]-->` | `Hanoi` | 0.1360 | 0.1088 | `[Candidate]` |
| **Hoan Kiem Lake** | `--[part_of]-->` | `Hanoi` | 0.1354 | 0.1083 | `[Candidate]` |
| **Ngoc Son Temple** | `--[belongs_to_dynasty]-->` | `Nguyen dynasty` | 0.1278 | 0.1022 | `[Candidate]` |
| **The Huc Bridge** | `--[has_color]-->` | `bright red` | 0.1240 | 0.0992 | `[Candidate]` |
| **Turtle Tower** | `--[situated_on]-->` | `Hoan Kiem Lake` | 0.1119 | 0.0895 | `[Candidate]` |
| **Ngoc Son Temple** | `--[located_in]-->` | `Hoan Kiem Lake` | 0.1049 | 0.0840 | `[Candidate]` |
| **Hoan Kiem Lake** | `--[located_in]-->` | `Hanoi` | 0.1022 | 0.0818 | `[Candidate]` |
| **traditional architectural style** | `--[belongs_to_dynasty]-->` | `Nguyen dynasty` | 0.1019 | 0.0815 | `[Candidate]` |

---

### Case 2: Van Mieu - Quoc Tu Giam (Hanoi)
**Văn bản nguồn:**
> "The Temple of Literature is located in Hanoi, Vietnam. The temple was built in 1070 by Emperor Ly Thanh Tong of the Ly dynasty. It is dedicated to Confucius. The complex was recognized as a special national relic in 2012."

**Đồ thị tri thức trích xuất (13 facts, 9 nodes):**

| Head | Quan hệ | Tail | Score (Gốc) | $\mu_F$ (Fuzzy) | Trạng thái |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Temple of Literature** | `--[built_in]-->` | `1070` | 0.4860 | 0.3888 | `[Caption OK]` |
| **Emperor Ly Thanh Tong** | `--[belongs_to_dynasty]-->` | `Ly dynasty` | 0.3452 | 0.2762 | `[Caption OK]` |
| **Hanoi** | `--[part_of]-->` | `Vietnam` | 0.3366 | 0.2693 | `[Caption OK]` |
| **Temple of Literature** | `--[dedicated_to]-->` | `Confucius` | 0.2948 | 0.2359 | `[Caption OK]` |
| **Temple of Literature** | `--[built_by]-->` | `Emperor Ly Thanh Tong` | 0.2845 | 0.2276 | `[Caption OK]` |
| **Hanoi** | `--[located_in]-->` | `Vietnam` | 0.2494 | 0.1996 | `[Caption OK]` |
| **Temple of Literature** | `--[located_in]-->` | `Vietnam` | 0.2315 | 0.1852 | `[Caption OK]` |
| **Temple of Literature** | `--[located_in]-->` | `Hanoi` | 0.2271 | 0.1817 | `[Caption OK]` |
| **Temple of Literature** | `--[recognized_as]-->` | `special national relic` | 0.2232 | 0.1786 | `[Caption OK]` |
| **Temple of Literature** | `--[recognized_in]-->` | `2012` | 0.1854 | 0.1484 | `[Candidate]` |
| **Temple of Literature** | `--[belongs_to_dynasty]-->` | `Ly dynasty` | 0.1558 | 0.1246 | `[Candidate]` |
| **Temple of Literature** | `--[part_of]-->` | `Vietnam` | 0.1274 | 0.1019 | `[Candidate]` |
| **Temple of Literature** | `--[dedicated_to]-->` | `Emperor Ly Thanh Tong` | 0.1063 | 0.0851 | `[Candidate]` |

---

### Case 3: Chua Thien Mu (Hue)
**Văn bản nguồn:**
> "Thien Mu Pagoda is an ancient pagoda situated in Hue along the Perfume River. The pagoda was built in 1601 by Nguyen Hoang of the Nguyen lords. It features Phuoc Duyen Tower and overlooks the river."

**Đồ thị tri thức trích xuất (5 facts, 6 nodes):**

| Head | Quan hệ | Tail | Score (Gốc) | $\mu_F$ (Fuzzy) | Trạng thái |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Nguyen Hoang** | `--[belongs_to_dynasty]-->` | `Nguyen lords` | 0.5916 | 0.4733 | `[Caption OK]` |
| **Thien Mu Pagoda** | `--[built_by]-->` | `Nguyen Hoang` | 0.4594 | 0.3675 | `[Caption OK]` |
| **Thien Mu Pagoda** | `--[built_in]-->` | `1601` | 0.3817 | 0.3053 | `[Caption OK]` |
| **Thien Mu Pagoda** | `--[located_in]-->` | `Hue` | 0.1774 | 0.1419 | `[Candidate]` |
| **Thien Mu Pagoda** | `--[situated_on]-->` | `Perfume River` | 0.1473 | 0.1178 | `[Candidate]` |

---

### Case 4: Pho Co Hoi An & Chua Cau (Quang Nam)
**Văn bản nguồn:**
> "Hoi An Ancient Town is an ancient trading port located in Quang Nam province, Vietnam. The town was recognized as a UNESCO World Heritage Site in 1999. The Japanese Covered Bridge, built in the 1590s by the Japanese community, crosses a canal connecting to the Thu Bon River. The bridge is made of wood in the traditional architectural style."

**Đồ thị tri thức trích xuất (11 facts, 9 nodes):**

| Head | Quan hệ | Tail | Score (Gốc) | $\mu_F$ (Fuzzy) | Trạng thái |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Quang Nam province** | `--[part_of]-->` | `Vietnam` | 0.4213 | 0.3370 | `[Caption OK]` |
| **Japanese Covered Bridge** | `--[made_of]-->` | `wood` | 0.3661 | 0.2929 | `[Caption OK]` |
| **Hoi An Ancient Town** | `--[recognized_as]-->` | `UNESCO World Heritage Site` | 0.3389 | 0.2711 | `[Caption OK]` |
| **Japanese Covered Bridge** | `--[architectural_style]-->` | `traditional architectural style` | 0.2582 | 0.2065 | `[Caption OK]` |
| **Quang Nam province** | `--[located_in]-->` | `Vietnam` | 0.2457 | 0.1966 | `[Caption OK]` |
| **Hoi An Ancient Town** | `--[located_in]-->` | `Vietnam` | 0.2200 | 0.1760 | `[Caption OK]` |
| **Hoi An Ancient Town** | `--[part_of]-->` | `Vietnam` | 0.2174 | 0.1739 | `[Caption OK]` |
| **Hoi An Ancient Town** | `--[located_in]-->` | `Quang Nam province` | 0.1792 | 0.1433 | `[Candidate]` |
| **Hoi An Ancient Town** | `--[recognized_in]-->` | `1999` | 0.1503 | 0.1203 | `[Candidate]` |
| **Japanese Covered Bridge** | `--[crosses]-->` | `Thu Bon River` | 0.1423 | 0.1139 | `[Candidate]` |
| **Hoi An Ancient Town** | `--[part_of]-->` | `Quang Nam province` | 0.1164 | 0.0931 | `[Candidate]` |

---

### Case 5: Vinh Ha Long (Quang Ninh)
**Văn bản nguồn:**
> "Ha Long Bay is a UNESCO World Heritage Site situated in Quang Ninh province, Vietnam. The bay features thousands of limestone islands in the Gulf of Tonkin. Sung Sot Cave, located on Bo Hon Island in the bay, was discovered by French explorers in 1901. Ti Top Island stands in the central part of the bay and is known for its crescent beach."

**Đồ thị tri thức trích xuất (14 facts, 8 nodes):**

| Head | Quan hệ | Tail | Score (Gốc) | $\mu_F$ (Fuzzy) | Trạng thái |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Sung Sot Cave** | `--[situated_on]-->` | `Bo Hon Island` | 0.3974 | 0.3179 | `[Caption OK]` |
| **Ha Long Bay** | `--[recognized_as]-->` | `UNESCO World Heritage Site` | 0.3755 | 0.3004 | `[Caption OK]` |
| **Quang Ninh province** | `--[part_of]-->` | `Vietnam` | 0.3579 | 0.2863 | `[Caption OK]` |
| **Bo Hon Island** | `--[located_in]-->` | `Ha Long Bay` | 0.3259 | 0.2607 | `[Caption OK]` |
| **Sung Sot Cave** | `--[located_in]-->` | `Ha Long Bay` | 0.2532 | 0.2026 | `[Caption OK]` |
| **Bo Hon Island** | `--[part_of]-->` | `Ha Long Bay` | 0.2436 | 0.1949 | `[Caption OK]` |
| **Ti Top Island** | `--[located_in]-->` | `Ha Long Bay` | 0.1912 | 0.1530 | `[Candidate]` |
| **Quang Ninh province** | `--[located_in]-->` | `Vietnam` | 0.1833 | 0.1467 | `[Candidate]` |
| **Ti Top Island** | `--[part_of]-->` | `Ha Long Bay` | 0.1727 | 0.1382 | `[Candidate]` |
| **Ha Long Bay** | `--[part_of]-->` | `Vietnam` | 0.1405 | 0.1124 | `[Candidate]` |
| **Ha Long Bay** | `--[located_in]-->` | `Gulf of Tonkin` | 0.1321 | 0.1057 | `[Candidate]` |
| **Ha Long Bay** | `--[located_in]-->` | `Vietnam` | 0.1163 | 0.0930 | `[Candidate]` |
| **Ha Long Bay** | `--[part_of]-->` | `Gulf of Tonkin` | 0.1104 | 0.0883 | `[Candidate]` |
| **Quang Ninh province** | `--[recognized_as]-->` | `UNESCO World Heritage Site` | 0.1098 | 0.0879 | `[Candidate]` |

---

### Case 6: Nha Tho Duc Ba (Ho Chi Minh City)
**Văn bản nguồn:**
> "Notre-Dame Cathedral of Saigon is a cathedral situated in Ho Chi Minh City, Vietnam. The cathedral was built in 1880 by French colonists in the Romanesque architectural style. The church was constructed of brick imported from France. It features two bell towers reaching a height of 58 m."

**Đồ thị tri thức trích xuất (9 facts, 6 nodes):**

| Head | Quan hệ | Tail | Score (Gốc) | $\mu_F$ (Fuzzy) | Trạng thái |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Notre-Dame Cathedral of Saigon** | `--[architectural_style]-->` | `Romanesque architectural style` | 0.4219 | 0.3375 | `[Caption OK]` |
| **Notre-Dame Cathedral of Saigon** | `--[made_of]-->` | `brick` | 0.2805 | 0.2244 | `[Caption OK]` |
| **Ho Chi Minh City** | `--[part_of]-->` | `Vietnam` | 0.2723 | 0.2178 | `[Caption OK]` |
| **Notre-Dame Cathedral of Saigon** | `--[located_in]-->` | `Vietnam` | 0.1458 | 0.1166 | `[Candidate]` |
| **Notre-Dame Cathedral of Saigon** | `--[built_in]-->` | `1880` | 0.1449 | 0.1159 | `[Candidate]` |
| **Ho Chi Minh City** | `--[located_in]-->` | `Vietnam` | 0.1367 | 0.1094 | `[Candidate]` |
| **Notre-Dame Cathedral of Saigon** | `--[part_of]-->` | `Vietnam` | 0.1342 | 0.1074 | `[Candidate]` |
| **Notre-Dame Cathedral of Saigon** | `--[located_in]-->` | `Ho Chi Minh City` | 0.1247 | 0.0998 | `[Candidate]` |
| **Notre-Dame Cathedral of Saigon** | `--[inaugurated_in]-->` | `1880` | 0.1009 | 0.0807 | `[Candidate]` |

---

### Case 7: Dai Noi Hue (Thua Thien Hue)
**Văn bản nguồn:**
> "The Imperial City of Hue is a walled palace situated in Hue along the Perfume River. The citadel was built in 1804 by Emperor Gia Long of the Nguyen dynasty. The complex was inscribed as a UNESCO World Heritage Site in 1993. The iconic Ngo Mon Gate, built in 1833, stands as the southern entrance to the palace."

**Đồ thị tri thức trích xuất (10 facts, 10 nodes):**

| Head | Quan hệ | Tail | Score (Gốc) | $\mu_F$ (Fuzzy) | Trạng thái |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Ngo Mon Gate** | `--[built_in]-->` | `1833` | 0.3481 | 0.2785 | `[Caption OK]` |
| **Imperial City of Hue** | `--[built_in]-->` | `1804` | 0.2857 | 0.2286 | `[Caption OK]` |
| **Imperial City of Hue** | `--[built_by]-->` | `Emperor Gia Long` | 0.2730 | 0.2184 | `[Caption OK]` |
| **Imperial City of Hue** | `--[recognized_as]-->` | `UNESCO World Heritage Site` | 0.2519 | 0.2015 | `[Caption OK]` |
| **Emperor Gia Long** | `--[belongs_to_dynasty]-->` | `Nguyen dynasty` | 0.2125 | 0.1700 | `[Caption OK]` |
| **Imperial City of Hue** | `--[recognized_in]-->` | `1993` | 0.1698 | 0.1358 | `[Candidate]` |
| **Imperial City of Hue** | `--[belongs_to_dynasty]-->` | `Nguyen dynasty` | 0.1689 | 0.1351 | `[Candidate]` |
| **Imperial City of Hue** | `--[located_in]-->` | `Hue` | 0.1294 | 0.1035 | `[Candidate]` |
| **Ngo Mon Gate** | `--[connects]-->` | `Imperial City of Hue` | 0.1097 | 0.0877 | `[Candidate]` |
| **Imperial City of Hue** | `--[situated_on]-->` | `Perfume River` | 0.1008 | 0.0807 | `[Candidate]` |

---

