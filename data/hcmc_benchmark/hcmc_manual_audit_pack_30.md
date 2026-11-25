# HCMC Manual Human Audit Pack (30 Stratified Cases, URI-Safe Rebuild)

This audit dossier pairs raw PySceneDetect video paths, blind `Qwen3-VL-8B` visual scene descriptions, grounding-independent Multi-VLM gold facts (`Indep-GT`), URI-safe VC-GAT input facts (`c_f`, `vcgat_score`), model captions, automated audit flags, and cross-family LLM judge verdicts so human reviewers can audit accuracy & abstention directly.

- Total cases: **30** (`{'BucketA_5way_MisID': 8, 'BucketB_35way_MisID': 6, 'BucketC_OAR_VSP_Audit': 6, 'BucketD_CorrectID_HIGH': 2, 'BucketD_CorrectID_LOW': 5, 'BucketD_CorrectID_MEDIUM': 3}`)

## Case 01 — `hcmc_0552` [BucketA_5way_MisID]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `1vK0TyASAIA` | **Tier:** `30s` | **p_onsite:** `0.982`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Du lịch Việt Nam： Dinh Độc Lập – di tích lịch sử quốc gia đặc biệ [1vK0TyASAIA]_scene_017.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.4905`, `LOW`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.1348`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an interior room with wooden-framed maps on the walls, a long wooden table with telephones, and a round table with ornate chairs and a floral centerpiece. The room has red curtains, patterned upholstery, and a visible doorway leading to another space.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, presidential reception halls)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9241, clip=0.983)
  - `(Independence Palace, has_part, entrance hall)` | mu_F=0.99049 | vis_R=None (qwen=0.0141, clip=0.9355)
  - `(Independence Palace, has_part, sun-shading louvers)` | mu_F=0.99049 | vis_R=None (qwen=0.0025, clip=0.8565)
  - `(Independence Palace, has_part, central flagpole)` | mu_F=0.99907 | vis_R=None (qwen=0.0001, clip=0.785)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, designed_by, Ngô Viết Thụ)` | mu_F=0.88484 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, located_in, District 1)` | mu_F=0.83493 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`37.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, a prominent structure in Ho Chi Minh City's District 1, stands as a testament to French colonial influence with its iconic arched windows and clock tower. The building's circular facade, featuring a Parisian-style clock, complements its ornate main hall and facade. Historical plaques commemorate the contributions of scientists and engineers. Nestled in Paris Commune Square, this landmark also houses an indoor history museum with curved glass vitrines displaying various exhibits. Despite the presence of other cultural displays within the complex, the post office remains a focal point for those interested in its architectural heritage and historical significance.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The museum interior displays a variety of exhibits, including a historic red banners exhibition, revolutionary war historic documents, and an indoor history museum with curved glass vitrines. Traditional silk tunics and artifacts related to motherhood and childbirth family rituals are also on display. The space features an imperial palace and a sacred inner shrine altar, though their exact placement within the room is unclear. Metal trays and tables are present, but their specific function remains undetermined. The overall atmosphere suggests a blend of historical significance and cultural heritage.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The museum interior features an exhibition of historic red banners alongside revolutionary war documents. Inside, there is a curved glass vitrine displaying various artifacts, including traditional silk tunics and items related to motherhood and childbirth. The space also includes an altar and an imperial palace model, though their exact identities are unclear.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The museum interior features an exhibition of historic red banners alongside revolutionary war documents. Inside, there is a curved glass vitrine displaying various artifacts, including traditional silk tunics and items related to motherhood and childbirth. The space also includes an altar and an imperial palace model, though their exact identities are unclear.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The museum interior features an exhibition of historic red banners alongside revolutionary war documents. Inside, there is a curved glass vitrine displaying various artifacts, including traditional silk tunics and items related to motherhood and childbirth. An altar stands nearby, though its exact purpose is unclear. The space also includes an imperial palace model and a sacred inner shrine altar, though their specific roles are not fully understood.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 02 — `hcmc_0417` [BucketA_5way_MisID]
- **GT Landmark:** `Ben Thanh Market` | **YouTube ID:** `ir3iI9apC7M` | **Tier:** `5s` | **p_onsite:** `0.9994`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Cho Ben Thanh/segment/Lịch Sử Chợ Bến Thành - Ben Thanh Market Ho Chi Minh City ｜ Vui T [ir3iI9apC7M]_scene_006.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.4454`, `LOW`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.1843`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows the exterior of a building with a prominent arched gateway and a clock tower above it, featuring visible signage that reads “CHỢ BẾN THÀNH” in gold letters. The structure has a beige facade with white trim, and the scene includes vehicles and pedestrians near the entrance under a partly cloudy sky.
- **Indep-GT Gold Facts:**
  - `(Ben Thanh Market, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9991, clip=0.9969)
  - `(Ben Thanh Market, located_in, Ho Chi Minh City)` | mu_F=0.99979 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`50.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, featuring arched windows, stands prominently in Ho Chi Minh City. Plaques commemorate scientists, adding to its cultural significance alongside the nearby colonial building and ancient stone citadel gate.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a mix of colonial architecture, including a yellow colonial museum facade and a yellow classical mansion estate, alongside traditional structures such as an ancient stone citadel gate and a pagoda. However, the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a colonial building, an ancient stone citadel gate, a yellow colonial museum facade, a tiered wooden bell tower, and other architectural elements such as a pagoda and a village gate.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a colonial building, an ancient stone citadel gate, a yellow colonial museum facade, a tiered wooden bell tower, and other architectural elements such as a pagoda and a village gate.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a colonial building, an ancient stone citadel gate, a yellow colonial museum facade, a tiered wooden bell tower, and other architectural elements such as a pagoda and a village gate.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Tie` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 03 — `hcmc_1090` [BucketA_5way_MisID]
- **GT Landmark:** `Saigon Opera House` | **YouTube ID:** `4ZU-XX5qrQo` | **Tier:** `5s` | **p_onsite:** `0.977`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Nha hat Thanh pho HCM/segment/KHUNG CAMHR NHÀ HÁT THÀNH PHỐ LÚC SÁNG SỚM#vietnam #travel #explo [4ZU-XX5qrQo]_scene_001.mp4`
- **Z4 5-way:** `Ben Thanh Market` (`mu_L=0.297`, `LOW`) | **Z4 35-way:** `Hanoi Old Quarter` (`mu_L=0.2162`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows a street scene with cars and pedestrians, featuring a mix of modern and classical architecture, including a prominent building with a curved glass facade and another with ornate white columns and arched windows. Visible signage includes a taxi with the number “38” and a green sign with “SUN” partially visible, alongside trees and a crosswalk in the foreground.
- **Indep-GT Gold Facts:**
  - `(Saigon Opera House, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9968, clip=0.7886)
  - `(Saigon Opera House, located_in, Ho Chi Minh City)` | mu_F=0.96933 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`50.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > Ben Thanh Market, featuring the southern clock tower and an interior hall, is located in Ho Chi Minh City. The market's surroundings include colonial buildings, a cyclo, a mosque, and a multi-tiered stone pavilion, alongside vibrant street vendors and motorbike traffic.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a mix of modern and traditional architectural elements, including a tube house and a colonial building, alongside a grand neoclassic theatre facade, a multi-tiered stone pavilion, and a church. Motorbikes and cyclos navigate the streets, while vendors sell goods from shoulder poles and a mosque stands nearby. The exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a tube house, a colonial building, motorbike traffic, and a grand neoclassic theatre facade. Nearby, there are cyclos, a church, and a multi-tiered stone pavilion. Shoulder pole vendors and motorbikes can also be seen.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`50.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`True`):
    > The scene features a tube house, a colonial building, motorbike traffic, and a grand neoclassic theatre facade. Nearby, there are cyclos, a church, and a multi-tiered stone pavilion. Shoulder pole vendors and motorbikes are also present.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a tube house, a colonial building, motorbike traffic, and a grand neoclassic theatre facade. Nearby, there are cyclos, a church, and a multi-tiered stone pavilion. Shoulder pole vendors and motorbikes can also be seen.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`B3_Prompt_Only_Certainty_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 04 — `hcmc_0273` [BucketA_5way_MisID]
- **GT Landmark:** `Ba Thien Hau Temple` | **YouTube ID:** `lJJV88ryLb0` | **Tier:** `5s` | **p_onsite:** `0.9996`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Chua Ba Thien Hau/segment/Khám phá Chùa Bà Thiên Hậu Q5-TPHCM mang vẻ đẹp uy nghiêm [lJJV88ryLb0]_scene_007.mp4`
- **Z4 5-way:** `Saigon Opera House` (`mu_L=0.4606`, `LOW`) | **Z4 35-way:** `Saigon Opera House` (`mu_L=0.2374`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows the ornate, multi-tiered roofline of a traditional Chinese temple, featuring intricate carvings, colorful decorative elements, and rows of ceramic roof tiles. A large crowd of people is visible on the upper levels of the temple structure, suggesting a public gathering or festival.
- **Indep-GT Gold Facts:**
  - `(Ba Thien Hau Temple, has_part, roof)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9985, clip=0.9812)
  - `(Ba Thien Hau Temple, located_in, Ho Chi Minh City)` | mu_F=0.98582 | vis_R=None (qwen=None, clip=None)
  - `(Ba Thien Hau Temple, dedicated_to, Mazu)` | mu_F=0.67792 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`66.7%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Opera House, located in Ho Chi Minh City, boasts a façade adorned with Flamboyant Beaux-Arts architecture. Its exterior features include a roof dragon ornament, carved stone dragons on the staircase, and a multi-tiered stone pavilion, complementing the traditional elements like the ancient stone citadel gate and sacred inner shrine altar.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a pagoda and a temple den, along with a carved stone dragon staircase and a multi-tiered stone pavilion. Dragon ornaments and wooden brackets can also be seen, but the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a pagoda, a temple den, and a multi-tiered stone pavilion. Nearby, there are carved stone dragons and a yellow colonial museum facade.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a pagoda, a temple den, and a multi-tiered stone pavilion. Nearby, there are carved stone dragons and a yellow colonial museum facade.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a pagoda, a temple den, and a multi-tiered stone pavilion. Nearby, there are carved stone dragons and a yellow colonial museum facade.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 05 — `hcmc_0553` [BucketA_5way_MisID]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `1vK0TyASAIA` | **Tier:** `30s` | **p_onsite:** `0.5622`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Du lịch Việt Nam： Dinh Độc Lập – di tích lịch sử quốc gia đặc biệ [1vK0TyASAIA]_scene_018.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.4048`, `LOW`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.0804`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an ornate interior room with red walls, a large wooden table, and decorative chairs. A framed painting of an eagle with outstretched wings hangs on the wall, and there are curtains, a fireplace, and wall-mounted lights visible.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, left wing)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9399, clip=0.9978)
  - `(Independence Palace, has_part, presidential reception halls)` | mu_F=0.99049 | vis_R=None (qwen=0.4378, clip=0.9847)
  - `(Independence Palace, has_part, entrance hall)` | mu_F=0.99049 | vis_R=None (qwen=0.023, clip=0.9962)
  - `(Independence Palace, has_part, central flagpole)` | mu_F=0.99907 | vis_R=None (qwen=0.0001, clip=0.9939)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, designed_by, Ngô Viết Thụ)` | mu_F=0.88484 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, located_in, District 1)` | mu_F=0.83493 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`25.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, a prominent structure in Ho Chi Minh City's District 1, stands as a testament to its architectural heritage. Its facade boasts arched windows and a circular clock tower, reflecting the Parisian style evident in its design. Historical maps from 1892 depict this building prominently, highlighting its significance in the city’s development. The post office's main hall and central clock tower are notable features, with plaques commemorating both scientists and engineers adorning its walls. Adjacent to the post office, there is an exhibition of historic military artillery, adding another layer to the complex's historical narrative. Surrounding the post office, one can also find exhibits on revolutionary war documents and artifacts, along with displays on motherhood and childbirth rituals, showcasing the diverse cultural and historical elements that coexist within the area.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The visual scene graph suggests a mix of historical artifacts and architectural elements, but the exact landmark identity remains uncertain. Historic red banners and revolutionary war documents hang alongside a phoenix motif, evoking a sense of rebirth and resilience. The museum interior features an imperial palace, old post office, war memorial, ancestral tablets, and exhibits on historic military artillery and family rituals. These diverse elements create a rich tapestry of history within the space.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The room features an exhibition of historic red banners and revolutionary war documents. There are also displays of a phoenix motif and other artifacts related to the imperial palace. The space includes an old post office and a war memorial, though their specific roles are unclear. Ancestral tablets and military artillery exhibits are present, but their exact meanings are not certain.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The room features an exhibition of historic red banners and revolutionary war documents. There are also displays of a phoenix motif and other artifacts related to the imperial palace. The space includes an old post office and a war memorial, though their specific roles are unclear. Ancestral tablets and military artillery exhibits are present, but their exact meanings are not certain.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The room features an exhibition of historic red banners, revolutionary war documents, and a phoenix motif. There are also displays related to the imperial palace, old post offices, war memorials, and ancestral tablets. Additionally, there is an exhibit on historic military artillery and information about motherhood and childbirth family rituals. The space appears to be part of a museum interior.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Tie` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B3_Prompt_Only_Certainty_5way` | vs B4 -> Qwen=`Tie`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 06 — `hcmc_0420` [BucketA_5way_MisID]
- **GT Landmark:** `Ben Thanh Market` | **YouTube ID:** `ir3iI9apC7M` | **Tier:** `10s` | **p_onsite:** `0.9707`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Cho Ben Thanh/segment/Lịch Sử Chợ Bến Thành - Ben Thanh Market Ho Chi Minh City ｜ Vui T [ir3iI9apC7M]_scene_009.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.5332`, `MEDIUM`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.1905`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** A large banner displays text commemorating the 110th anniversary of Ben Thanh Market (1914–2024), featuring an image of a clock tower and surrounding buildings. The banner is mounted on a wooden-framed structure with visible roof tiles above, and the text includes both Vietnamese and English translations.
- **Indep-GT Gold Facts:**
  - `(Ben Thanh Market, has_part, roofs)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9981, clip=0.9972)
  - `(Ben Thanh Market, has_part, roof)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9959, clip=0.9947)
  - `(Ben Thanh Market, located_in, Ho Chi Minh City)` | mu_F=0.99979 | vis_R=None (qwen=None, clip=None)
  - `(Ben Thanh Market, built_in, early 1900s)` | mu_F=0.83871 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Saigon Central Post Office, on, Cong Xa Paris Square)` | c_f=0.3413 | mu_F=0.64019 | vcgat=10.806
  - `(design, architectural_style, Indo-Saracenic architecture)` | c_f=0.114 | mu_F=0.21376 | vcgat=10.7094
  - `(Saigon Central Post Office, located_in, Ho Chi Minh City)` | c_f=0.5063 | mu_F=0.94963 | vcgat=10.6131
  - `(Saigon Central Post Office, located_in, District 1)` | c_f=0.4475 | mu_F=0.83925 | vcgat=10.356
  - `(plaques, commemorates, engineers)` | c_f=0.2398 | mu_F=0.44976 | vcgat=9.1977
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`25.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, an iconic colonial building in District 1 of Ho Chi Minh City, features arched windows and a prominent clock tower. Plaques commemorate local scientists, while the surrounding area includes a colonial building, a village gate, and various cultural displays such as a historic red banners exhibition and a conical palm hat artistic display.
  - **B3_Prompt_Only_Certainty_5way** (regime=`MEDIUM`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene likely shows a structure that appears to be the Saigon Central Post Office, located on Cong Xa Paris Square in District 1 of Ho Chi Minh City. Indo-Saracenic architecture is evident, and plaques commemorate the engineers who designed it. If this is indeed the Saigon Central Post Office, it stands as a prominent example of colonial-era buildings in the area.
  - **B4_Ours_No_VCGAT_5way** (regime=`MEDIUM`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, likely showing its arched windows and clock tower, stands prominently in District 1 of Ho Chi Minh City. Nearby, there are hints of colonial architecture and other cultural elements, though their exact identification is uncertain.
  - **Proposed_VCGAT_E2E_5way** (regime=`MEDIUM`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, if this is indeed Saigon Central Post Office, appears to be an Indo-Saracenic architectural marvel located in Ho Chi Minh City's District 1, surrounded by colonial buildings and other cultural elements such as prayer flags and tiered wooden bell towers.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a tube house, a colonial building, a prayer flag banner, a pagoda, a tiered wooden bell tower, historic red banners on display, a church, text, conical palm hats, and a village gate. However, the specific landmark identity is unclear.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`B4_Ours_No_VCGAT_5way`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 07 — `hcmc_1091` [BucketA_5way_MisID]
- **GT Landmark:** `Saigon Opera House` | **YouTube ID:** `4ZU-XX5qrQo` | **Tier:** `10s` | **p_onsite:** `0.9985`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Nha hat Thanh pho HCM/segment/KHUNG CAMHR NHÀ HÁT THÀNH PHỐ LÚC SÁNG SỚM#vietnam #travel #explo [4ZU-XX5qrQo]_scene_002.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.3186`, `LOW`) | **Z4 35-way:** `The Government Guest House` (`mu_L=0.1322`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an ornate, light-colored building with arched windows and decorative stonework, situated on a busy urban street with modern glass buildings nearby. Pedestrians and vehicles are visible, along with green banners and trees lining the sidewalk under a partly cloudy sky.
- **Indep-GT Gold Facts:**
  - `(Saigon Opera House, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9985, clip=0.9831)
  - `(Saigon Opera House, has_part, theatre façade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.989, clip=0.9529)
  - `(Saigon Opera House, located_in, Ho Chi Minh City)` | mu_F=0.96933 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Opera House, located_in, Vietnam)` | mu_F=0.8117 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`50.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, an iconic colonial building in District 1 of Ho Chi Minh City, features arched windows and a prominent clock tower. Plaques commemorate scientists, adding to its rich history. Surrounding the post office, one can see a mix of architectural styles including a grand neoclassic theatre facade, a yellow classical mansion estate, and a traditional temple courtyard sanctuary.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The visual scene includes a colonial building, a grand neoclassic theatre facade, a tube house, a yellow colonial museum facade, a multi-tiered stone pavilion, a church, a yellow classical mansion estate, a historic red banners exhibition, a mosque, and a traditional temple courtyard sanctuary. However, the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a colonial building, a grand neoclassic theatre facade, a tube house, and a yellow colonial museum facade. There is also a multi-tiered stone pavilion and a church visible, along with a traditional temple courtyard sanctuary and historic red banners on display. However, the exact landmark identity remains uncertain.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a colonial building, a grand neoclassic theatre facade, a tube house, and a yellow colonial museum facade. There is also a multi-tiered stone pavilion and a church visible, along with a traditional temple courtyard sanctuary and historic red banners on display. However, the exact landmark identity remains uncertain.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`25.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`True`):
    > The scene features a colonial building, a grand neoclassic theatre facade, a tube house, and a yellow colonial museum facade. There is also a multi-tiered stone pavilion and a church visible, along with other less certain structures like a yellow classical mansion estate and historic red banners.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B3_Prompt_Only_Certainty_5way` | vs B4 -> Qwen=`Tie`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 08 — `hcmc_0614` [BucketA_5way_MisID]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `UiN-ZFOj2zk` | **Tier:** `30s` | **p_onsite:** `0.9998`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Dinh độc Lập - Những bí mật lịch sử [UiN-ZFOj2zk]_scene_055.mp4`
- **Z4 5-way:** `Ba Thien Hau Temple` (`mu_L=0.3485`, `LOW`) | **Z4 35-way:** `Ba Thien Hau Temple` (`mu_L=0.1208`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows a large, ornate room with a red carpet featuring intricate gold patterns, flanked by dark wooden chairs and tables. A central table holds a vase with bright yellow flowers, and a red curtain with a golden emblem is visible in the background.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, presidential reception halls)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.989, clip=0.9804)
  - `(Independence Palace, has_part, entrance hall)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9627, clip=0.9646)
  - `(Independence Palace, has_part, ground plan)` | mu_F=0.99049 | vis_R=None (qwen=0.5622, clip=0.3969)
  - `(Independence Palace, has_part, Trung)` | mu_F=0.9025 | vis_R=None (qwen=0.0019, clip=0.6957)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, designed_by, Ngô Viết Thụ)` | mu_F=0.88484 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, located_in, District 1)` | mu_F=0.83493 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`12.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > Ba Thien Hau Temple, an integral part of Cholon's cultural landscape in Ho Chi Minh City, Vietnam, stands as a testament to the region's rich heritage. The temple features upper walls, a courtyard, and an altar dedicated to Mazu, the revered sea goddess. Its distinctive Yin-Yang tiled roofs and constructed with material wood, it harmoniously blends traditional Vietnamese architecture with elements of Cantonese influence. Adjacent to the temple, the Tuệ Thành Assembly Hall showcases Chinese-style architecture, offering a juxtaposition of styles within the same vicinity. Visitors can observe the sacred inner shrine altar, traditional silk tunic exhibitions, and other cultural displays, creating a vibrant tapestry of history and art.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The visual scene graph suggests a mix of cultural and historical elements, but the exact landmark identity remains uncertain. The imperial palace and the sacred inner shrine altar are visible, along with the historic red banners exhibition. Traditional silk tunic and wedding costume displays are also present, alongside a monastery hall and a lacquered column. Additionally, there are exhibits featuring glazed ceramic porcelain vases and an artistic display of conical palm hats. The yellow classical mansion estate and other architectural parts are part of this complex, though their specific roles within the site are unclear.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an imperial palace, a sacred inner shrine altar, historic red banners on display, a monastery hall, and lacquered columns. There is also a traditional silk tunic exhibition and a yellow classical mansion estate. Additionally, there are glazed ceramic porcelain vases and conical palm hats displayed artistically. The atmosphere suggests a cultural heritage site or museum.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an imperial palace, a sacred inner shrine altar, historic red banners on display, a monastery hall, and lacquered columns. There is also a traditional silk tunic exhibition and a yellow classical mansion estate. Additionally, there are glazed ceramic porcelain vases and conical palm hats displayed artistically. The atmosphere suggests a cultural heritage site or museum.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an imperial palace, a sacred inner shrine altar, and a historic red banners exhibition. Nearby, there is a monastery hall and a lacquered column. Traditional silk tunics and wedding costumes are displayed, along with glazed ceramic porcelain vases and conical palm hats. However, the specific identity of the main landmark remains unclear.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 09 — `hcmc_0132` [BucketB_35way_MisID]
- **GT Landmark:** `Ba Thien Hau Temple` | **YouTube ID:** `ebNzfxPO_Hg` | **Tier:** `10s` | **p_onsite:** `0.9998`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Chua Ba Thien Hau/segment/Chùa Bà Thiên Hậu - Chợ Lớn [ebNzfxPO_Hg]_scene_049.mp4`
- **Z4 5-way:** `Ba Thien Hau Temple` (`mu_L=0.6736`, `MEDIUM`) | **Z4 35-way:** `Thang Long Water Puppetry Theatre` (`mu_L=0.222`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an outdoor temple entrance with a large, ornate wooden gate adorned with hanging golden decorations and red vertical banners with white text. Numerous incense sticks burn in front of the gate, emitting smoke, and a crowd of people, many wearing face masks, gathers in front of the structure.
- **Indep-GT Gold Facts:**
  - `(Ba Thien Hau Temple, has_part, faces)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.977, clip=0.9919)
  - `(Ba Thien Hau Temple, has_part, main structure)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9399, clip=0.9865)
  - `(Ba Thien Hau Temple, located_in, Ho Chi Minh City)` | mu_F=0.98582 | vis_R=None (qwen=None, clip=None)
  - `(Ba Thien Hau Temple, located_in, Cholon)` | mu_F=0.75061 | vis_R=None (qwen=None, clip=None)
  - `(Ba Thien Hau Temple, dedicated_to, Mazu)` | mu_F=0.67792 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Ba Thien Hau Temple, has_part, altar)` | c_f=0.6735 | mu_F=0.99991 | vcgat=11.0087
  - `(Ba Thien Hau Temple, architectural_style, Vietnamese architecture)` | c_f=0.1576 | mu_F=0.234 | vcgat=9.9467
  - `(Ba Thien Hau Temple, founded_in, 1760)` | c_f=0.2562 | mu_F=0.38041 | vcgat=10.8586
  - `(Ba Thien Hau Temple, dated_to, 1760)` | c_f=0.1785 | mu_F=0.26498 | vcgat=10.8476
  - `(Ba Thien Hau Temple, dedicated_to, Mazu)` | c_f=0.4566 | mu_F=0.67792 | vcgat=11.0331
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`60.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > Ba Thien Hau Temple, located in Cholon, Ho Chi Minh City, features an upper wall and a courtyard. Inside, one can see a sacred inner shrine altar, incense sticks, and an incense burner. Adjacent to the temple, there is a multi-tiered red Buddhist stupa, adding to the serene atmosphere of this dedicated space for Mazu.
  - **B3_Prompt_Only_Certainty_5way** (regime=`MEDIUM`, K-Rec=`20.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene likely shows a structure possibly belonging to Ba Thien Hau Temple, which appears to feature an altar and Vietnamese architecture. Founded in 1760, it is dedicated to Mazu and includes elements such as incense coils, incense sticks, and a sacred inner shrine altar. If this is indeed Ba Thien Hau Temple, it also features a multi-tiered red Buddhist stupa and a dragon dance performance nearby.
  - **B4_Ours_No_VCGAT_5way** (regime=`MEDIUM`, K-Rec=`60.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Ba Thien Hau Temple, likely showing its upper walls and courtyard, appears to be located in Cholon, Ho Chi Minh City. Inside, there are incense coils and an altar dedicated to Mazu, though other elements like conical palm hats and water puppet wooden dolls are also present.
  - **Proposed_VCGAT_E2E_5way** (regime=`MEDIUM`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Ba Thien Hau Temple likely shows an altar and exhibits Vietnamese architectural elements. If this is indeed Ba Thien Hau Temple, it may also feature incense coils and sacred inner shrines.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an incense coil, a sacred inner shrine altar, conical palm hats, incense sticks, water puppet wooden dolls on water, a temple den, a pagoda, a dragon dance, an incense burner, and a multi-tiered red Buddhist stupa. However, the exact landmark identity remains uncertain.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B3_Prompt_Only_Certainty_5way` | vs B4 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 10 — `hcmc_0135` [BucketB_35way_MisID]
- **GT Landmark:** `Ba Thien Hau Temple` | **YouTube ID:** `ebNzfxPO_Hg` | **Tier:** `10s` | **p_onsite:** `0.9997`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Chua Ba Thien Hau/segment/Chùa Bà Thiên Hậu - Chợ Lớn [ebNzfxPO_Hg]_scene_052.mp4`
- **Z4 5-way:** `Ba Thien Hau Temple` (`mu_L=0.4872`, `LOW`) | **Z4 35-way:** `Hanoi Cuisine` (`mu_L=0.4838`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an outdoor temple courtyard with a large incense burner emitting smoke, surrounded by people holding incense sticks. Visible signage in the lower right corner reads “Chùa Bà Thiên Hậu” and “Chợ Lớn,” indicating the location.
- **Indep-GT Gold Facts:**
  - `(Ba Thien Hau Temple, has_part, Lanterns)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9241, clip=0.9386)
  - `(Ba Thien Hau Temple, has_part, courtyard)` | mu_F=0.99991 | vis_R=1.0 (qwen=0.9859, clip=0.7733)
  - `(Ba Thien Hau Temple, located_in, Ho Chi Minh City)` | mu_F=0.98582 | vis_R=None (qwen=None, clip=None)
  - `(Ba Thien Hau Temple, located_in, Cholon)` | mu_F=0.75061 | vis_R=None (qwen=None, clip=None)
  - `(Ba Thien Hau Temple, dedicated_to, Mazu)` | mu_F=0.67792 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`60.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > Ba Thien Hau Temple, located in Cholon, Ho Chi Minh City, features an upper wall and a courtyard. Incense coils, sticks, and bundles are burned at the site, while a sacred inner shrine altar stands within the temple. Nearby, a shoulder pole vendor offers goods, and a conical palm hat is artistically displayed, creating a vibrant atmosphere around the temple.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The visual scene features an incense burner with red incense bundles, incense sticks, and an incense coil. A small sacred inner shrine altar and a conical palm hat artistic display are also present, along with a shoulder pole vendor. Steam rises from nearby food, but the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an incense burner, incense coils, incense sticks, and incense bundles of various colors. Nearby, there is an artistic display of conical palm hats and a small sacred inner shrine altar. Steam rises from food nearby, though the specific type of food is unclear.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`True`):
    > The scene features an incense burner, incense coils, incense sticks, and incense bundles of various colors. Nearby, there is an artistic display of conical palm hats and a small sacred inner shrine altar. Steam rises from food nearby, though the specific type of food is not clearly visible.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`True`):
    > The scene features an incense burner, incense coils, incense sticks, and incense bundles of various colors. Nearby, there is an artistic display of conical palm hats and a small sacred inner shrine altar. Steam rises from food nearby, though the specific type of food is not clear.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 11 — `hcmc_0259` [BucketB_35way_MisID]
- **GT Landmark:** `Ba Thien Hau Temple` | **YouTube ID:** `ieTMxXsyJdY` | **Tier:** `5s` | **p_onsite:** `0.9941`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Chua Ba Thien Hau/segment/Chùa Bà Thiên Hậu (miếu Đức Bà) Quận 5 P.1 [ieTMxXsyJdY]_scene_006.mp4`
- **Z4 5-way:** `Ba Thien Hau Temple` (`mu_L=0.5389`, `MEDIUM`) | **Z4 35-way:** `Thang Long Water Puppetry Theatre` (`mu_L=0.1726`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an ornate, traditional East Asian-style gate with intricate golden carvings and decorative teal metalwork framing the entrance. Above the gate, a detailed wooden relief depicts numerous figures and mythical creatures, with red tassels hanging from the structure.
- **Indep-GT Gold Facts:**
  - `(Ba Thien Hau Temple, has_part, iron gate)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.989, clip=0.9059)
  - `(Ba Thien Hau Temple, located_in, Ho Chi Minh City)` | mu_F=0.98582 | vis_R=None (qwen=None, clip=None)
  - `(Ba Thien Hau Temple, dedicated_to, Mazu)` | mu_F=0.67792 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Ba Thien Hau Temple, has_part, altar)` | c_f=0.5389 | mu_F=0.99991 | vcgat=11.1719
  - `(Ba Thien Hau Temple, founded_by, immigrants)` | c_f=0.2428 | mu_F=0.45048 | vcgat=10.4375
  - `(altar, dedicated_to, Goddess)` | c_f=0.1081 | mu_F=0.20056 | vcgat=11.1363
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`66.7%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > Ba Thien Hau Temple, located in Ho Chi Minh City, is dedicated to Mazu and features upper walls. Inside, intricate dragon carvings adorn the staircases, while guardian lions guard the entrance, and a sacred inner shrine altar stands at the center. Lanterns line the streets, creating a serene atmosphere.
  - **B3_Prompt_Only_Certainty_5way** (regime=`MEDIUM`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > If this is indeed Ba Thien Hau Temple, the scene likely shows an altar dedicated to the Goddess, possibly indicating its religious significance, alongside other architectural elements such as dragon-carved stairs and a lantern street.
  - **B4_Ours_No_VCGAT_5way** (regime=`MEDIUM`, K-Rec=`33.3%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Ba Thien Hau Temple, if this is indeed Ba Thien Hau Temple, features upper walls and appears to be dedicated to Mazu. Nearby, there are water puppet wooden dolls on water and a dragon carved staircase.
  - **Proposed_VCGAT_E2E_5way** (regime=`MEDIUM`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Ba Thien Hau Temple likely shows an altar dedicated to the Goddess, though it also features other elements such as water puppet wooden dolls on water and dragon-carved stairs.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a sacred inner shrine altar, water puppet wooden dolls on water, a temple den, dragon-carved stairs, a pagoda, guardian lion statues, a maternal rituals altar, roof dragon ornaments, three entrance gates, and a lantern street. However, the exact landmark identity remains uncertain.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Tie` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Tie` | vs B4 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 12 — `hcmc_0265` [BucketB_35way_MisID]
- **GT Landmark:** `Ba Thien Hau Temple` | **YouTube ID:** `ieTMxXsyJdY` | **Tier:** `5s` | **p_onsite:** `0.9941`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Chua Ba Thien Hau/segment/Chùa Bà Thiên Hậu (miếu Đức Bà) Quận 5 P.1 [ieTMxXsyJdY]_scene_012.mp4`
- **Z4 5-way:** `Ba Thien Hau Temple` (`mu_L=0.4371`, `LOW`) | **Z4 35-way:** `Hanoi Cuisine` (`mu_L=0.1005`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows a traditional Chinese-style gate with red pillars and dark wooden beams, leading into an open courtyard with a large incense burner and a person seated on a bench in the background. A woman in a white shirt and red bag walks through the gate, while another person in a white shirt and black cap walks past on the right side.
- **Indep-GT Gold Facts:**
  - `(Ba Thien Hau Temple, has_part, courtyard)` | mu_F=0.99991 | vis_R=None (qwen=0.9987, clip=0.2242)
  - `(Ba Thien Hau Temple, located_in, Ho Chi Minh City)` | mu_F=0.98582 | vis_R=None (qwen=None, clip=None)
  - `(Ba Thien Hau Temple, dedicated_to, Mazu)` | mu_F=0.67792 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`100.0%`, FactP=`100.0%`, VSP_VLM=`0.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > Ba Thien Hau Temple, dedicated to Mazu, features upper walls and a traditional temple courtyard sanctuary within Ho Chi Minh City. The scene includes a sacred inner shrine altar, traditional silk tunic exhibition, and sidewalk street food stalls alongside a conical hat artistic display.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`33.3%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The temple den and traditional temple courtyard sanctuary are visible, along with a sacred inner shrine altar and a traditional silk tunic exhibition. A conical palm hat artistic display and sidewalk street food stalls can also be seen, alongside a shoulder pole vendor and a tube house. The presence of these elements suggests a blend of modern and traditional elements within the area, though the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`33.3%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a traditional temple courtyard sanctuary, a sacred inner shrine altar, and a conical palm hat artistic display among other cultural exhibits and street food stalls.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`33.3%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a traditional temple courtyard sanctuary, a sacred inner shrine altar, and a conical palm hat artistic display among other cultural exhibits and street food stalls.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`33.3%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a temple den, a sacred inner shrine altar, a traditional silk tunic exhibition, and a conical palm hat artistic display. Nearby, there are sidewalk street food stalls and a traditional temple courtyard sanctuary.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B3_Prompt_Only_Certainty_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 13 — `hcmc_0293` [BucketB_35way_MisID]
- **GT Landmark:** `Ben Thanh Market` | **YouTube ID:** `AGhll-2UgwQ` | **Tier:** `10s` | **p_onsite:** `0.9988`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Cho Ben Thanh/segment/🛍️ Từ Chợ Bến Thành đến Takashimaya ｜ Khám phá Sài Gòn từ truyền [AGhll-2UgwQ]_scene_001.mp4`
- **Z4 5-way:** `Ben Thanh Market` (`mu_L=0.6998`, `MEDIUM`) | **Z4 35-way:** `Hanoi Old Quarter` (`mu_L=0.2745`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video opens with an exterior shot of a large, yellowish-brown colonial-style building featuring arched entrances, a central clock tower, and red-tiled roofing, with visible signage reading “CITY ELECTRONICS” and “BÁN HÀNG LIÊN” above the arches. Inside a bustling market, vendors display goods on red plastic stools under a high, wooden-beamed ceiling with hanging lights, surrounded by crowds of shoppers and colorful merchandise.
- **Indep-GT Gold Facts:**
  - `(Ben Thanh Market, has_part, market hall)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9933, clip=0.9404)
  - `(Ben Thanh Market, has_part, interior)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9968, clip=0.9126)
  - `(Ben Thanh Market, located_in, Ho Chi Minh City)` | mu_F=0.99979 | vis_R=None (qwen=None, clip=None)
  - `(Ben Thanh Market, built_in, early 1900s)` | mu_F=0.83871 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Ben Thanh Market, part_of, city)` | c_f=0.3137 | mu_F=0.44828 | vcgat=10.8243
  - `(Ben Thanh Market, part_of, the city)` | c_f=0.1952 | mu_F=0.27888 | vcgat=10.7721
  - `(Ben Thanh Market, founded_in, 1914)` | c_f=0.4833 | mu_F=0.69069 | vcgat=11.0478
  - `(Ben Thanh Market, built_by, Valentin de La Croix)` | c_f=0.4132 | mu_F=0.59049 | vcgat=11.0286
  - `(Ben Thanh Market, architectural_style, French-style)` | c_f=0.5102 | mu_F=0.72904 | vcgat=10.6297
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`100.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > In the heart of Ho Chi Minh City, Ben Thanh Market stands as a testament to its early 1900s construction, featuring a southern clock tower and an interior hall with a domed roof. The market bustles with activity, showcasing vibrant displays of woven silk fabric bolts, colorful silk lanterns hanging overhead, and traditional conical palm hats on display. Alongside these elements, colonial buildings and tube houses blend harmoniously with the modern night market, creating a unique atmosphere.
  - **B3_Prompt_Only_Certainty_5way** (regime=`MEDIUM`, K-Rec=`0.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > The bustling evening pedestrian market likely shows Ben Thanh Market, which appears to be a colonial building featuring French-style architecture. If this is indeed Ben Thanh Market, it features colorful silk lanterns and a colorful canopy walkway, suggesting its origins date back to 1914 when it was built by Valentin de La Croix.
  - **B4_Ours_No_VCGAT_5way** (regime=`MEDIUM`, K-Rec=`75.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Ben Thanh Market, likely showing its southern clock tower and interior hall, stands prominently in Ho Chi Minh City. It features a domed roof and colonial architecture, though some elements such as colorful silk lanterns and woven silk fabric bolts displays suggest it may also serve as a bustling evening pedestrian market.
  - **Proposed_VCGAT_E2E_5way** (regime=`MEDIUM`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Ben Thanh Market, if this is indeed Ben Thanh Market, features a mix of bustling evening pedestrian activity, including sidewalk street food stalls and woven silk fabric bolts displays. The market also showcases French-style architecture, evident in its colonial building and colorful silk lanterns.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a bustling evening pedestrian market with colorful silk lanterns hanging overhead, woven silk fabric bolts displayed on stalls, and sidewalk street food stalls. Conical palm hats are also on display, creating an artistic focal point. Nearby, there are colonial buildings and tube houses, though the exact landmark identity remains uncertain.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`Tie` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Tie` | vs B4 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Tie`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 14 — `hcmc_0296` [BucketB_35way_MisID]
- **GT Landmark:** `Ben Thanh Market` | **YouTube ID:** `AGhll-2UgwQ` | **Tier:** `20s` | **p_onsite:** `0.9933`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Cho Ben Thanh/segment/🛍️ Từ Chợ Bến Thành đến Takashimaya ｜ Khám phá Sài Gòn từ truyền [AGhll-2UgwQ]_scene_004.mp4`
- **Z4 5-way:** `Ben Thanh Market` (`mu_L=0.3944`, `LOW`) | **Z4 35-way:** `Lake Thien Quang` (`mu_L=0.1146`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an outdoor urban plaza with a prominent yellow and red building featuring a large clock tower and the visible text “QUYEN TRANH” on its facade. Surrounding the plaza are modern high-rise buildings, palm trees, benches, pedestrians, and a large digital billboard displaying colorful advertisements.
- **Indep-GT Gold Facts:**
  - `(Ben Thanh Market, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9627, clip=0.9414)
  - `(Ben Thanh Market, has_part, market hall)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9707, clip=0.9459)
  - `(Ben Thanh Market, has_part, roofs)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9999, clip=0.8375)
  - `(Ben Thanh Market, located_in, Ho Chi Minh City)` | mu_F=0.99979 | vis_R=None (qwen=None, clip=None)
  - `(Ben Thanh Market, built_in, early 1900s)` | mu_F=0.83871 | vis_R=None (qwen=None, clip=None)
  - `(Ben Thanh Market, designed_by, Valentin de La Croix)` | mu_F=0.81427 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`50.0%`, FactP=`0.0%`, VSP_VLM=`40.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > Ben Thanh Market, a prominent landmark in Ho Chi Minh City, features a southern clock tower and an ornate interior hall with a domed roof. Built in the early 1900s, it stands as a testament to colonial architecture, showcasing ornate columns and a main hall. The market's presence is notable amidst the diverse architectural styles of Ho Chi Minh City, including a colonial building, a skyscraper, and a multi-tiered stone pavilion. Surrounding the market, one can observe a mix of traditional and modern elements, such as a conical palm hat artistic display and park amusement carnival rides.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The visual scene features a mix of architectural styles, including a yellow classical mansion estate, a colonial building, and a skyscraper. Nearby, there is a multi-tiered stone pavilion and a courtyard. A conical palm hat artistic display adds a touch of color, while a tube house and a stele house provide additional structures. The presence of park amusement carnival rides suggests a recreational area, but the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an artistic display of conical palm hats, a yellow classical mansion estate, and a colonial building. Nearby, there are skyscrapers and tube houses, along with a park with amusement carnival rides. Additionally, there is a multi-tiered stone pavilion and a conical hat visible. The area also includes a courtyard, though its specific identity is unclear.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an artistic display of conical palm hats, a yellow classical mansion estate, and a colonial building. Nearby, there are skyscrapers and tube houses, along with a park with amusement carnival rides. Additionally, there is a multi-tiered stone pavilion and a conical hat visible. The area also includes a courtyard, though its specific identity is unclear.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an artistic display of conical palm hats, a yellow classical mansion estate, and a colonial building. Nearby, there are skyscrapers and tube houses, along with a park with amusement carnival rides. Additionally, there is a multi-tiered stone pavilion and a conical hat visible. The area also includes a courtyard, though its specific identity is unclear.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 15 — `hcmc_0848` [BucketC_OAR_VSP_Audit]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `V2ET9GKNu6A` | **Tier:** `10s` | **p_onsite:** `0.9968`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Tham quan và khám phá bên trong Dinh Độc Lập [V2ET9GKNu6A]_scene_132.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.2723`, `LOW`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.0943`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an ornate interior hall with red carpeting featuring a large yellow Chinese character, flanked by rows of patterned armchairs and small tables. Large windows draped with yellow curtains allow natural light, and decorative red panels with gold patterns are visible on the walls.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, entrance hall)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.974, clip=0.994)
  - `(Independence Palace, has_part, presidential reception halls)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9627, clip=0.9962)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`25.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, with its arched windows and clock tower, stands prominently in District 1 of Ho Chi Minh City. Nearby, the old post office and traditional silk tunic exhibition provide a contrast to the historical significance of the area, which also features exhibits on historic red banners and traditional wedding costumes.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The visual scene includes elements such as an imperial palace, a traditional silk tunic exhibition, a monastery hall, a sacred inner shrine altar, a historic red banners exhibition, an old post office, a museum interior, a traditional wedding costume display, a yellow classical mansion estate, and a traditional bridal costumes mannequin display. However, the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene features an imperial palace, a traditional silk tunic exhibition, a monastery hall, and a sacred inner shrine altar. There are also historic red banners on display, an old post office, and various exhibits including traditional wedding costumes and bridal costumes. The area appears to be part of a museum interior.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene features an imperial palace, a traditional silk tunic exhibition, a monastery hall, and a sacred inner shrine altar. There are also historic red banners on display, an old post office, and various exhibits related to traditional clothing and culture.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene features an imperial palace, a traditional silk tunic exhibition, a monastery hall, and a sacred inner shrine altar. There are also historic red banners on display, an old post office, and various exhibits including traditional wedding costumes and bridal costumes. The area appears to be part of a museum interior.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 16 — `hcmc_0638` [BucketC_OAR_VSP_Audit]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `UiN-ZFOj2zk` | **Tier:** `10s` | **p_onsite:** `0.9997`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Dinh độc Lập - Những bí mật lịch sử [UiN-ZFOj2zk]_scene_079.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.4433`, `LOW`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.1702`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an opulent interior room with red carpeting, ornate wooden chairs with red upholstery, and large windows draped with golden curtains. A mirrored wall reflects the room’s furnishings, and a glass-top table with a dark wooden base is visible in the foreground.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, presidential reception halls)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9933, clip=0.9935)
  - `(Independence Palace, has_part, entrance hall)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.8808, clip=0.9815)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`25.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, with its arched windows and clock tower, stands prominently in District 1 of Ho Chi Minh City. Plaques commemorate scientists within its walls, while the building's architectural features contrast with the nearby traditional bridal costumes mannequin display and the historic red banners exhibition.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The visual scene includes elements such as an imperial palace, a yellow classical mansion estate, a historic red banners exhibition, a traditional silk tunic exhibition, a sacred inner shrine altar, an old post office, a museum interior, a monastery hall, a lacquered column, and a traditional bridal costumes mannequin display. However, the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene features an imperial palace, a yellow classical mansion estate, a historic red banners exhibition, and a traditional silk tunic exhibition. Nearby, there is an old post office and a museum interior with lacquered columns. Traditional bridal costumes are displayed on mannequins.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene features an imperial palace, a yellow classical mansion estate, a historic red banners exhibition, and a traditional silk tunic exhibition. Nearby, there is an old post office and a museum interior with lacquered columns. Traditional bridal costumes are displayed on mannequins.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene features an imperial palace, a yellow classical mansion estate, a historic red banners exhibition, and a traditional silk tunic exhibition. Nearby, there is an old post office and a museum interior with lacquered columns. Traditional bridal costumes are displayed on mannequins.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 17 — `hcmc_0818` [BucketC_OAR_VSP_Audit]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `V2ET9GKNu6A` | **Tier:** `10s` | **p_onsite:** `0.977`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Tham quan và khám phá bên trong Dinh Độc Lập [V2ET9GKNu6A]_scene_102.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.6289`, `MEDIUM`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.171`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an indoor room with wooden-paneled walls and a patterned tiled floor, featuring a large framed map and a banner above the door that reads “PHÒNG HỒ TỔNG THỐNG” with English text below. A small red sign with a white symbol is visible on a stand near the map, and the ceiling has fluorescent lighting fixtures.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, ground plan)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9941, clip=0.9704)
  - `(Independence Palace, has_part, Trung)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.6225, clip=0.9581)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Saigon Central Post Office, has_part, plaques)` | c_f=0.5676 | mu_F=0.9025 | vcgat=11.546
  - `(Saigon Central Post Office, has_part, interior walls)` | c_f=0.5676 | mu_F=0.9025 | vcgat=10.9899
  - `(Saigon Central Post Office, built_by, French architects)` | c_f=0.1757 | mu_F=0.27944 | vcgat=11.0839
  - `(Saigon Central Post Office, built_by, engineer)` | c_f=0.1321 | mu_F=0.21 | vcgat=10.713
  - `(plaques, commemorates, engineers)` | c_f=0.2829 | mu_F=0.44976 | vcgat=11.502
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`25.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, a prominent landmark in Ho Chi Minh City's District 1, features arched windows and a clock tower. Plaques commemorate scientists within its walls. The scene captures the post office’s interior, showcasing historic red banners, revolutionary war documents, and a curved glass vitrine displaying artifacts.
  - **B3_Prompt_Only_Certainty_5way** (regime=`MEDIUM`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > If this is indeed Saigon Central Post Office, it likely shows plaques commemorating engineers and interior walls adorned with historical documents. The building appears to have been constructed by French architects and an engineer, and it possibly features historic red banners and revolutionary war documents within its interior.
  - **B4_Ours_No_VCGAT_5way** (regime=`MEDIUM`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office likely shows arched windows and a clock tower, though it is not certain. Inside, there are exhibits of historic red banners and revolutionary war documents, though these may not exclusively belong to this building.
  - **Proposed_VCGAT_E2E_5way** (regime=`MEDIUM`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office likely shows plaques and interior walls, though its exact structure is less clear. Nearby, there are historic red banners on display and an indoor history museum with a curved glass vitrine.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The space features an exhibition of historic red banners, a museum interior, and revolutionary war documents. Inside, there are curved glass vitrines and a motherhood and childbirth family ritual display. Modern glass exhibits and conical palm hats are also present. The exact landmark identity remains unknown.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Tie` | vs B4 -> Qwen=`B4_Ours_No_VCGAT_5way`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 18 — `hcmc_0794` [BucketC_OAR_VSP_Audit]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `V2ET9GKNu6A` | **Tier:** `5s` | **p_onsite:** `0.982`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Tham quan và khám phá bên trong Dinh Độc Lập [V2ET9GKNu6A]_scene_078.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.4412`, `LOW`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.1628`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows a spacious, well-lit interior room with high ceilings featuring exposed wooden beams and recessed lighting. Large windows draped with heavy brown curtains allow natural light to enter, and the floor is tiled in light beige, complemented by dark wood furniture and decorative elements like framed artwork on the walls.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, entrance hall)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9669, clip=0.8725)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`50.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, featuring arched windows, stands prominently in Ho Chi Minh City. Commemorative plaques honor scientists, alongside other cultural exhibits within the historic building.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The visual scene includes elements such as the imperial palace, old post office, museum interior, communal house, and monastery hall, along with traditional silk tunic exhibitions, wooden pillars, and sacred inner shrine altars. However, the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene features an imperial palace, an old post office, a museum interior, a communal house, a monastery hall, a traditional silk tunic exhibition, wooden pillars, a sacred inner shrine altar, and other structures.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene features an imperial palace, an old post office, a museum interior, a communal house, a monastery hall, a traditional silk tunic exhibition, wooden pillars, a sacred inner shrine altar, and other structures.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene features an imperial palace, an old post office, a museum interior, a communal house, a monastery hall, a traditional silk tunic exhibition, a wooden pillar, a sacred inner shrine altar, and other structures.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`Tie` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 19 — `hcmc_0763` [BucketC_OAR_VSP_Audit]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `V2ET9GKNu6A` | **Tier:** `20s` | **p_onsite:** `0.9859`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Tham quan và khám phá bên trong Dinh Độc Lập [V2ET9GKNu6A]_scene_047.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.318`, `LOW`) | **Z4 35-way:** `Huu Tiep Lake` (`mu_L=0.0774`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an interior room with red carpeting, wooden furniture, and large framed artwork on the walls. Red curtains frame windows, and a group of people is visible walking through the space.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, presidential reception halls)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9399, clip=0.9123)
  - `(Independence Palace, has_part, entrance hall)` | mu_F=0.99049 | vis_R=None (qwen=0.3208, clip=0.9288)
  - `(Independence Palace, has_part, Trung)` | mu_F=0.9025 | vis_R=None (qwen=0.0373, clip=0.848)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, designed_by, Ngô Viết Thụ)` | mu_F=0.88484 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`16.7%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, a prominent landmark in Ho Chi Minh City's District 1, features arched windows and a clock tower on its facade. Historical maps from 1892 and plaques commemorating scientists and engineers are also part of the building’s rich history. The current scene showcases an exhibition within the post office, featuring traditional silk tunics, bridal costumes, and documents related to the revolutionary war.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The museum interior features a mix of exhibits, including a historic red banners exhibition, a traditional silk tunic display, and a mannequin display of traditional bridal costumes. Additionally, there is a traditional wedding costume display. The space also includes archival documents related to the revolutionary war and a monastery hall. Other notable structures visible nearby include an old post office and a motherhood and childbirth family rituals exhibit. The exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The museum interior features an exhibition of historic red banners alongside displays of traditional silk tunics and bridal costumes. Nearby, there are mannequins showing traditional wedding costumes, and documents related to the revolutionary war are kept in archives. The space also includes a monastery hall and exhibits on motherhood and childbirth family rituals. However, the specific landmark identity remains unclear.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The museum interior features an exhibition of historic red banners alongside displays of traditional silk tunics and bridal costumes. Nearby, there are mannequins showing traditional wedding costumes, and documents related to the revolutionary war are also on display. The space also includes an old post office and a monastery hall, though their specific roles within this exhibit are unclear.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`True`):
    > The space features an exhibition of historic red banners, traditional silk tunics, and bridal costumes displayed on mannequins. There are also displays of traditional wedding costumes and revolutionary war documents. The room has an air of cultural heritage, though the specific landmark cannot be identified with certainty.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`B4_Ours_No_VCGAT_5way`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 20 — `hcmc_0706` [BucketC_OAR_VSP_Audit]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `UiN-ZFOj2zk` | **Tier:** `30s` | **p_onsite:** `0.9996`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Dinh độc Lập - Những bí mật lịch sử [UiN-ZFOj2zk]_scene_147.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.512`, `MEDIUM`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.1725`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows the exterior of a neoclassical-style building with arched doorways and columns, featuring visible signage in both Vietnamese and English that reads “TƯ DINH NORODOM ĐẾN DINH ĐỘC LẬP 1868—1966” and “The exhibit 'From Norodom Palace to Independence Palace 1868—1966'”. The building has a light-colored facade, blue-tinted glass windows, and a red flag with a yellow emblem mounted above
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9975, clip=0.996)
  - `(Independence Palace, has_part, Trung)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9959, clip=0.9859)
  - `(Independence Palace, has_part, white facade)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9996, clip=0.9786)
  - `(Independence Palace, has_part, entrance hall)` | mu_F=0.99049 | vis_R=None (qwen=0.2227, clip=0.9856)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, designed_by, Ngô Viết Thụ)` | mu_F=0.88484 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, located_in, District 1)` | mu_F=0.83493 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Saigon Central Post Office, has_part, interior walls)` | c_f=0.4621 | mu_F=0.9025 | vcgat=11.407
  - `(Saigon Central Post Office, has_part, design)` | c_f=0.4621 | mu_F=0.9025 | vcgat=11.357
  - `(Saigon Central Post Office, has_part, plaques)` | c_f=0.4621 | mu_F=0.9025 | vcgat=11.2933
  - `(design, architectural_style, Indo-Saracenic architecture)` | c_f=0.1094 | mu_F=0.21376 | vcgat=11.1241
  - `(Saigon Central Post Office, built_by, French architects)` | c_f=0.1431 | mu_F=0.27944 | vcgat=10.7687
  - `(Saigon Central Post Office, located_in, District 1)` | c_f=0.4297 | mu_F=0.83925 | vcgat=10.7193
  - `(historical maps, dated_to, 1892)` | c_f=0.1677 | mu_F=0.32752 | vcgat=10.6837
  - `(Saigon Central Post Office, located_in, Paris Commune Square)` | c_f=0.1565 | mu_F=0.30568 | vcgat=10.6717
  - `(plaques, commemorates, engineers)` | c_f=0.2303 | mu_F=0.44976 | vcgat=11.2695
  - `(plaques, commemorates, scientists)` | c_f=0.232 | mu_F=0.45312 | vcgat=11.0163
  - `(circular facade clock, architectural_style, Parisian-style)` | c_f=0.1492 | mu_F=0.29136 | vcgat=5.2127
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`37.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, a prominent landmark in Ho Chi Minh City's District 1, stands as a testament to colonial architecture. Its facade boasts an intricate circular design with a Parisian-inspired clock tower, while arched windows and a main hall complete its structure. Historical maps from 1892 depict its location in Paris Commune Square. Plaques within the post office commemorate both scientists and engineers, adding to its cultural significance. Adjacent to this historic building, the scene captures a mix of colonial-era elements, including a yellow colonial museum facade and a traditional silk tunic exhibition, alongside more modern displays like a historic red banners exhibition and a tube house.
  - **B3_Prompt_Only_Certainty_5way** (regime=`MEDIUM`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The scene likely shows a colonial building that appears to be the Saigon Central Post Office, possibly located in District 1 at Paris Commune Square. The structure's facade is adorned with yellow colonial museum facades and traditional silk tunic exhibitions, suggesting it is a significant historical site. If this is indeed Saigon Central Post Office, it features Indo-Saracenic architectural elements, including plaques that commemorate both engineers and scientists. The building's interior walls and design are notable, and a circular facade clock exhibits Parisian-style architecture.
  - **B4_Ours_No_VCGAT_5way** (regime=`MEDIUM`, K-Rec=`37.5%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, likely showing an arched window and a clock tower, stands prominently in the District 1 area of Ho Chi Minh City. Its facade, if this is indeed the Saigon Central Post Office, features a circular design with a Parisian-style architecture. Nearby, there are historic red banners on display and a yellow colonial museum facade, though these may not directly belong to the post office itself. The site also includes a traditional silk tunic exhibition and a conical palm hat artistic display, suggesting a rich cultural heritage.
  - **Proposed_VCGAT_E2E_5way** (regime=`MEDIUM`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, likely showing its Indo-Saracenic architectural style, stands prominently in District 1. Inside, the building features interior walls and design elements that hint at its historical significance. Plaques commemorating engineers and scientists are visible, though their exact placement is uncertain. The building's exterior, if this is indeed the Saigon Central Post Office, displays a yellow colonial museum facade and a circular facade with a Parisian-style clock tower.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`12.5%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a colonial building with a yellow facade, an exhibition of historic red banners, and a yellow classical mansion estate. Nearby, there is a tube house and a traditional silk tunic exhibition. Additionally, there is an artistic display of conical palm hats and a church. The area also includes a monastery hall and a temple den, though their specific identities are unclear.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 21 — `hcmc_1058` [BucketD_CorrectID_HIGH]
- **GT Landmark:** `Saigon Central Post Office` | **YouTube ID:** `Y2zYPfCo8Js` | **Tier:** `20s` | **p_onsite:** `0.9993`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Buu dien trung tam Sai Gon/segment/BƯU ĐIỆN TRUNG TÂM SÀI GÒN - NÉT CỔ KÍNH GIỮA LÒNG THÀNH PHỐ HIỆN [Y2zYPfCo8Js]_scene_008.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.8408`, `HIGH`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.4516`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** A large, arched wall mural depicts a detailed map of the Atlantic coast, framed by classical architectural elements including columns and ornate moldings. Below the mural, a row of decorative clocks with red star accents is mounted on a dark wooden panel, flanked by glass display cases and illuminated signage.
- **Indep-GT Gold Facts:**
  - `(Saigon Central Post Office, has_part, interior)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9999, clip=0.9992)
  - `(Saigon Central Post Office, has_part, historical maps)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9991, clip=0.9999)
  - `(Saigon Central Post Office, has_part, main hall)` | mu_F=0.99907 | vis_R=1.0 (qwen=0.977, clip=0.9862)
  - `(Saigon Central Post Office, located_in, Ho Chi Minh City)` | mu_F=0.94963 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Central Post Office, located_in, District 1)` | mu_F=0.83925 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Central Post Office, designed_by, Jules Bourard)` | mu_F=0.74855 | vis_R=None (qwen=None, clip=None)
  - `(plaques, commemorates, scientists)` | mu_F=0.45312 | vis_R=None (qwen=None, clip=None)
  - `(plaques, commemorates, engineers)` | mu_F=0.44976 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Saigon Central Post Office, has_part, plaques)` | c_f=0.7588 | mu_F=0.9025 | vcgat=11.0187
  - `(Saigon Central Post Office, on, Cong Xa Paris Square)` | c_f=0.5383 | mu_F=0.64019 | vcgat=10.9006
  - `(Saigon Central Post Office, has_part, historical maps)` | c_f=0.7588 | mu_F=0.9025 | vcgat=10.7128
  - `(Saigon Central Post Office, built_by, French architects)` | c_f=0.235 | mu_F=0.27944 | vcgat=10.7204
  - `(historical maps, dated_to, 1892)` | c_f=0.2754 | mu_F=0.32752 | vcgat=10.7085
  - `(Saigon Central Post Office, built_by, engineer)` | c_f=0.1766 | mu_F=0.21 | vcgat=10.67
  - `(plaques, commemorates, engineers)` | c_f=0.3782 | mu_F=0.44976 | vcgat=10.9965
  - `(plaques, commemorates, scientists)` | c_f=0.381 | mu_F=0.45312 | vcgat=10.8984
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`62.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, a prominent landmark in Ho Chi Minh City's District 1, features arched windows and a clock tower on its facade. Historical maps from 1892 depict its location. Plaques commemorate both scientists and engineers. The scene juxtaposes the historic post office with modern exhibitions, including a traditional silk tunic display and a red banner exhibition.
  - **B3_Prompt_Only_Certainty_5way** (regime=`HIGH`, K-Rec=`37.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Saigon Central Post Office stands prominently on Cong Xa Paris Square, featuring plaques that commemorate both engineers and scientists. Historical maps dating back to 1892 are also displayed within the building, adding to its historical significance. The structure was constructed by French architects and an engineer, reflecting the colonial influence on Vietnamese architecture. Adjacent to the post office, there is a historic red banners exhibition, hinting at the broader cultural and historical context of the area.
  - **B4_Ours_No_VCGAT_5way** (regime=`HIGH`, K-Rec=`37.5%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, located in District 1 of Ho Chi Minh City, features an arched windowed facade and a prominent clock tower. Nearby, there is an exhibition of historic red banners and a museum interior, though these are not part of the main landmark itself. The building also includes elements such as a traditional silk tunic exhibition and a sacred inner shrine altar, though their exact locations are not clearly visible.
  - **Proposed_VCGAT_E2E_5way** (regime=`HIGH`, K-Rec=`37.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Saigon Central Post Office stands prominently on Cong Xa Paris Square, featuring plaques commemorating engineers and scientists. Historical maps dating back to 1892 are also displayed here, adding to its rich historical significance. Nearby, there are exhibits of historic red banners and a traditional silk tunic display, though these do not belong to this specific landmark.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an exhibition of historic red banners, an old post office, and parts of the imperial palace. There is also a sacred inner shrine altar and a traditional silk tunic exhibition. Nearby, there is a river and a tube house. Parallel sentence boards are visible, though their purpose is unclear. The overall atmosphere suggests a cultural heritage site, but the specific landmark identity remains uncertain.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B3_Prompt_Only_Certainty_5way` | vs B4 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 22 — `hcmc_0561` [BucketD_CorrectID_HIGH]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `UiN-ZFOj2zk` | **Tier:** `30s` | **p_onsite:** `0.9999`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Dinh độc Lập - Những bí mật lịch sử [UiN-ZFOj2zk]_scene_002.mp4`
- **Z4 5-way:** `Independence Palace` (`mu_L=0.8149`, `HIGH`) | **Z4 35-way:** `Ba Dinh Square` (`mu_L=0.1726`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** A man stands in a large green lawn in front of a large, multi-story white building with a central flagpole and red flags on either side. The building has a symmetrical facade with multiple windows and a flag at the top, under an overcast sky with trees visible on the periphery.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9975, clip=0.9958)
  - `(Independence Palace, has_part, white facade)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9999, clip=0.9949)
  - `(Independence Palace, has_part, beautiful garden)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.982, clip=0.9818)
  - `(Independence Palace, has_part, central flagpole)` | mu_F=0.99907 | vis_R=1.0 (qwen=0.9841, clip=0.9641)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, designed_by, Ngô Viết Thụ)` | mu_F=0.88484 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, located_in, District 1)` | mu_F=0.83493 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Independence Palace, part_of, Republic of Vietnam)` | c_f=0.1908 | mu_F=0.23416 | vcgat=11.1666
  - `(Independence Palace, architectural_style, Baroque Revival)` | c_f=0.2416 | mu_F=0.29648 | vcgat=10.532
  - `(Independence Palace, part_of, Ho Chi Minh City)` | c_f=0.169 | mu_F=0.20744 | vcgat=10.3406
  - `(Independence Palace, surrounded_by, Ho Chi Minh City)` | c_f=0.4108 | mu_F=0.50416 | vcgat=10.3305
  - `(Independence Palace, built_by, Ngô Đình Diệm)` | c_f=0.4502 | mu_F=0.55247 | vcgat=11.2035
  - `(Independence Palace, commissioned_by, Republic of Vietnam)` | c_f=0.2554 | mu_F=0.31336 | vcgat=11.198
  - `(Independence Palace, commissioned_by, Ngô Đình Diệm)` | c_f=0.5724 | mu_F=0.70238 | vcgat=11.1975
  - `(Independence Palace, built_by, Ngô Viết Thụ)` | c_f=0.2486 | mu_F=0.30512 | vcgat=11.0977
  - `(Independence Palace, architectural_style, French-style)` | c_f=0.6252 | mu_F=0.76726 | vcgat=10.301
  - `(Independence Palace, part_of, city centre)` | c_f=0.3418 | mu_F=0.41947 | vcgat=10.1527
  - `(Independence Palace, has_part, facade)` | c_f=0.8072 | mu_F=0.99049 | vcgat=8.7217
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`62.5%`, FactP=`100.0%`, VSP_VLM=`66.7%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Independence Palace stands prominently in the heart of Ho Chi Minh City's bustling District 1, its imposing facade adorned with traditional temple courtyards and a multi-tiered stone pavilion. The central flagpole, flanked by colonial building elements and surrounded by an expansive manicured lawn, greets visitors. Adjacent to the main structure, a historic red banners exhibition and a conical palm hat artistic display offer a glimpse into the region’s cultural heritage. The rooftop pavilion and sun-shading louvers provide shade for state banquets and presidential receptions held within the grand grounds. This architectural masterpiece, constructed by the French colonial authorities, is a testament to the city's rich history and serves as a focal point for national ceremonies.
  - **B3_Prompt_Only_Certainty_5way** (regime=`HIGH`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Independence Palace, prominently situated within the city centre of Ho Chi Minh City, is a striking example of Baroque Revival architecture. Built by Ngô Đình Diệm and commissioned by the Republic of Vietnam, it stands as a testament to French colonial influence, featuring a façade adorned with intricate details. Surrounded by expansive manicured lawns and a colonial building, the palace includes a traditional temple courtyard and a national flagpole guarded by military personnel. Its presence in the heart of Ho Chi Minh City underscores its significance as a symbol of the country's past.
  - **B4_Ours_No_VCGAT_5way** (regime=`HIGH`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`50.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Independence Palace stands prominently in Ho Chi Minh City's District 1, its central flagpole clearly visible. The palace features a ground plan with sun-shading louvers and a rooftop pavilion, typical of French colonial architecture. It houses state banquet areas and presidential reception halls, showcasing its importance as a symbol of national sovereignty. Surrounding the palace is an expansive manicured green lawn, though nearby there are hints of other structures like a yellow classical mansion estate and a traditional temple courtyard sanctuary.
  - **Proposed_VCGAT_E2E_5way** (regime=`HIGH`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Independence Palace, located in the heart of Ho Chi Minh City, features a prominent Baroque Revival style facade. Surrounding it is an expansive manicured green lawn, while nearby stands a colonial building. The palace itself is a significant landmark, constructed by Ngô Đình Diệm and commissioned by the Republic of Vietnam. Its imposing structure includes a multi-tiered stone pavilion and a historic brick flag tower, contributing to its rich cultural heritage.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a yellow classical mansion estate surrounded by an expansive manicured green lawn. Nearby, there is a colonial building and a traditional temple courtyard sanctuary. Historic red banners and conical palm hats are displayed, along with an imperial palace and a multi-tiered stone pavilion. A historic brick flag tower stands nearby, though its exact purpose is unclear. Military guards are positioned at a national flagpole, adding to the sense of tradition and authority.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`B4_Ours_No_VCGAT_5way`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 23 — `hcmc_1074` [BucketD_CorrectID_LOW]
- **GT Landmark:** `Saigon Central Post Office` | **YouTube ID:** `Y2zYPfCo8Js` | **Tier:** `30s` | **p_onsite:** `0.9859`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Buu dien trung tam Sai Gon/segment/BƯU ĐIỆN TRUNG TÂM SÀI GÒN - NÉT CỔ KÍNH GIỮA LÒNG THÀNH PHỐ HIỆN [Y2zYPfCo8Js]_scene_024.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.4762`, `LOW`) | **Z4 35-way:** `Hanoi Cuisine` (`mu_L=0.2009`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows the interior of a large, ornate building with high arched ceilings, exposed wooden beams, and multiple levels of seating or booths. People are seated at tables or standing in groups, with visible signage including a blue sign reading “WOMEN’S HEALTH CLINIC” and other signs indicating “CASH” and “CASHIER.”
- **Indep-GT Gold Facts:**
  - `(Saigon Central Post Office, has_part, interior)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9985, clip=0.9431)
  - `(Saigon Central Post Office, has_part, arched windows)` | mu_F=0.99991 | vis_R=1.0 (qwen=0.9981, clip=0.7897)
  - `(Saigon Central Post Office, has_part, ornamental pilasters)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9399, clip=0.8496)
  - `(Saigon Central Post Office, has_part, main hall)` | mu_F=0.99907 | vis_R=1.0 (qwen=0.989, clip=0.7353)
  - `(Saigon Central Post Office, located_in, Ho Chi Minh City)` | mu_F=0.94963 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Central Post Office, located_in, District 1)` | mu_F=0.83925 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Central Post Office, designed_by, Jules Bourard)` | mu_F=0.74855 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Central Post Office, designed_by, Auguste Henri Vildieu)` | mu_F=0.7433 | vis_R=None (qwen=None, clip=None)
  - `(plaques, commemorates, scientists)` | mu_F=0.45312 | vis_R=None (qwen=None, clip=None)
  - `(plaques, commemorates, engineers)` | mu_F=0.44976 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`40.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, a prominent colonial-era structure located in Ho Chi Minh City's District 1, stands as a testament to the city's rich history. Its circular facade, adorned with Parisian-style architecture, features an intricate clock tower and arched windows. Historical maps from 1892 depict its presence in the area. The post office's main hall and facade are notable architectural elements. Adjacent to it, a bustling evening pedestrian market and street food stalls create a lively atmosphere, contrasting with the timeless elegance of the post office. Tourists often gather around the site, appreciating both the historical significance and the vibrant local culture.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene unfolds on a bustling evening pedestrian market, where the atmosphere is alive with the sounds of street food stalls and the chatter of a tourist group. Amidst the vibrant mix of colonial buildings and tube houses, an old post office stands as a relic of the past. Market stalls and street food carts dot the sidewalks, creating a lively and diverse environment that captures the essence of the area's vibrant community. Despite the presence of these elements, the exact landmark identity remains uncertain, leaving the specific historical significance of this location open to interpretation.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a bustling evening pedestrian market with street food stalls and a tourist group. Nearby, there are old post offices and colonial buildings, though their specific identities are unclear. Tube houses and communal houses can also be seen, contributing to the local atmosphere.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a bustling evening pedestrian market with street food stalls and a tourist group. Nearby, there are old post offices and colonial buildings, though their specific identities are unclear. Tube houses and communal houses can also be seen, contributing to the local atmosphere.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a bustling evening pedestrian market with street food stalls and a tourist group. Nearby, there are old post offices and colonial buildings, though their specific identities are unclear. Tube houses and communal houses can also be seen, contributing to the local atmosphere.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`B3_Prompt_Only_Certainty_5way` | vs B4 -> Qwen=`Tie`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 24 — `hcmc_0538` [BucketD_CorrectID_LOW]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `1vK0TyASAIA` | **Tier:** `30s` | **p_onsite:** `0.989`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Du lịch Việt Nam： Dinh Độc Lập – di tích lịch sử quốc gia đặc biệ [1vK0TyASAIA]_scene_003.mp4`
- **Z4 5-way:** `Independence Palace` (`mu_L=0.3934`, `LOW`) | **Z4 35-way:** `Le Nin Park` (`mu_L=0.0665`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows a modern building with white and beige architectural elements, including a covered walkway and stone-textured walls, surrounded by greenery. A golden plaque with visible text is displayed, and people are seen walking near the entrance under a bright sky.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, white facade)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9981, clip=0.9162)
  - `(Independence Palace, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9047, clip=0.9577)
  - `(Independence Palace, has_part, beautiful garden)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.8808, clip=0.9284)
  - `(Independence Palace, has_part, sun-shading louvers)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.6792, clip=0.9948)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, designed_by, Ngô Viết Thụ)` | mu_F=0.88484 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, located_in, District 1)` | mu_F=0.83493 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Independence Palace stands prominently in the heart of Ho Chi Minh City's bustling District 1, its imposing facade adorned with traditional silk tunic exhibitions and a calligraphy inscribed wooden plaque. The central flagpole and sun-shading louvers enhance its grandeur, while the expansive manicured green lawn provides a serene backdrop for state banquets and presidential receptions. Adjacent to the palace, a historic red banners exhibition and a colonial building offer glimpses into the past, complementing the modern architecture of the Independence Palace itself.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`True`, LowStyleOnly=`False`):
    > The visual scene features an imperial palace with an inscribed wooden plaque, an expansive manicured green lawn, a traditional silk tunic exhibition, a museum interior, an old post office, a historic red banners exhibition, a colonial building, a moon gate, and various texts. The exact landmark identity remains uncertain, as no definitive information was provided.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`True`):
    > The scene features an expansive manicured green lawn surrounded by an imperial palace-like structure. Nearby, there is a calligraphy-inscribed wooden plaque and a traditional silk tunic exhibition. The area also includes a museum interior and a historic red banners exhibition. Additionally, there are colonial buildings and a moon gate visible in the background.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an expansive manicured green lawn, an imperial palace, and a calligraphy-inscribed wooden plaque. Nearby, there is a traditional silk tunic exhibition and a museum interior. Additionally, there are historic red banners on display and a colonial building nearby. The area also includes a moon gate and some text, though the specific meaning of these elements is unclear.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an imperial palace, an expansive manicured green lawn, and a calligraphy-inscribed wooden plaque. Nearby, there is a traditional silk tunic exhibition and a museum interior. Additionally, there are historic red banners on display and a colonial building nearby. The area also includes a moon gate and some text, though the specific meaning of these elements is unclear.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`B4_Ours_No_VCGAT_5way`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 25 — `hcmc_0297` [BucketD_CorrectID_LOW]
- **GT Landmark:** `Ben Thanh Market` | **YouTube ID:** `AGhll-2UgwQ` | **Tier:** `20s` | **p_onsite:** `0.9933`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Cho Ben Thanh/segment/🛍️ Từ Chợ Bến Thành đến Takashimaya ｜ Khám phá Sài Gòn từ truyền [AGhll-2UgwQ]_scene_005.mp4`
- **Z4 5-way:** `Ben Thanh Market` (`mu_L=0.4547`, `LOW`) | **Z4 35-way:** `Hanoi Old Quarter` (`mu_L=0.1226`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows an outdoor plaza with a large yellow building featuring a clock tower and red signage reading “CÔNG VIÊN THÀNH” above an arched entrance. People are walking or sitting on benches, and modern buildings with billboards and palm trees are visible in the background under an overcast sky.
- **Indep-GT Gold Facts:**
  - `(Ben Thanh Market, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.977, clip=0.9807)
  - `(Ben Thanh Market, has_part, market hall)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9526, clip=0.9786)
  - `(Ben Thanh Market, has_part, roofs)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9997, clip=0.9129)
  - `(Ben Thanh Market, located_in, Ho Chi Minh City)` | mu_F=0.99979 | vis_R=None (qwen=None, clip=None)
  - `(Ben Thanh Market, built_in, early 1900s)` | mu_F=0.83871 | vis_R=None (qwen=None, clip=None)
  - `(Ben Thanh Market, designed_by, Valentin de La Croix)` | mu_F=0.81427 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`33.3%`, FactP=`0.0%`, VSP_VLM=`50.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > Ben Thanh Market, an iconic colonial-era structure in Ho Chi Minh City, Vietnam, features a southern clock tower and an ornate interior hall with a domed roof. Ornate columns adorn the exterior, and the main hall stands as a testament to its architectural heritage. The market's vibrant atmosphere is evident with vendors displaying conical hats and street food carts, alongside a tourist group exploring the area. Surrounding structures include a yellow classical mansion and traditional tube houses, creating a picturesque setting that reflects the city's rich history.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The visual scene includes a shoulder pole vendor, a conical palm hat artistic display, a colonial building, street food carts, a tourist group, sidewalk street food stalls, a conical hat, a yellow classical mansion estate, a cyclo, and tube houses. The exact landmark identity remains uncertain, as no additional information was provided.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a colonial building, a street food cart, and sidewalk street food stalls. Nearby, there is a shoulder pole vendor and an artistic display of conical palm hats. Tourists can be seen, though their exact location is unclear. Yellow classical mansions and tube houses are also present, adding to the local atmosphere.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a colonial building, a street food cart, and sidewalk street food stalls. Nearby, there is a shoulder pole vendor and an artistic display of conical palm hats. Tourists can be seen, though their exact location is unclear. Yellow classical mansions and tube houses are also present, adding to the local atmosphere.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a shoulder pole vendor selling goods, an artistic display of conical palm hats, a colonial building, street food carts, a tourist group, sidewalk street food stalls, conical hats, and a yellow classical mansion estate. Cyclos can also be seen nearby.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`Tie` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 26 — `hcmc_1098` [BucketD_CorrectID_LOW]
- **GT Landmark:** `Saigon Opera House` | **YouTube ID:** `MR0nlPLVw8Q` | **Tier:** `30s` | **p_onsite:** `0.9998`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Nha hat Thanh pho HCM/segment/ĐỘC LẠ NHÀ HÁT THÀNH PHỐ HỒ CHÍ MINH [MR0nlPLVw8Q]_scene_004.mp4`
- **Z4 5-way:** `Saigon Opera House` (`mu_L=0.3095`, `LOW`) | **Z4 35-way:** `The Government Guest House` (`mu_L=0.1127`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows the exterior of a grand, light-colored building with classical architecture, featuring tall columns, ornate carvings, and multiple steps leading to the entrance. Yellow flowers in planters adorn the steps, and a green sign with white text reading “Bảo tàng Lịch sử” is visible on the right side of the building.
- **Indep-GT Gold Facts:**
  - `(Saigon Opera House, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9997, clip=0.9806)
  - `(Saigon Opera House, has_part, façade)` | mu_F=1.0 | vis_R=1.0 (qwen=0.9997, clip=0.9701)
  - `(Saigon Opera House, has_part, theatre façade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.989, clip=0.9169)
  - `(Saigon Opera House, has_part, décor)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9047, clip=0.9917)
  - `(Saigon Opera House, located_in, Ho Chi Minh City)` | mu_F=0.96933 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Opera House, located_in, Vietnam)` | mu_F=0.8117 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Opera House, located_in, District 1)` | mu_F=0.74189 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Opera House, designed_by, Paul Bourret)` | mu_F=0.65443 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`62.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Saigon Opera House, a prominent landmark in Ho Chi Minh City, Vietnam, stands as a testament to architectural beauty. Its façade, featuring Flamboyant Beaux-Arts architecture, is adorned with carved stone dragons and a grand arched central portal. The building's interior includes a symmetrical tower and a raised floor, enhancing the grandeur of its auditorium and theatre façade. Located in District 1, it was constructed by Daydé & Pillat. Adjacent to the opera house, a historic red banners exhibition can be found, adding cultural depth to the area. Surrounding structures include a colonial building and a multi-tiered stone pavilion, creating a rich tapestry of historical and architectural elements.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`12.5%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The visual scene graph depicts a diverse array of architectural elements, each contributing to the rich tapestry of the area's history. A tube house stands out amidst the colonial building and the grand neoclassic theatre facade, while a carved stone dragon staircase adds an intriguing touch. The yellow classical mansion estate and the historic red banners exhibition provide a glimpse into the past, alongside the yellow colonial museum facade. Historical significance is further underscored by the historic brick flag tower and the multi-tiered stone pavilion. However, the exact landmark identity remains uncertain, as no definitive information is provided.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`12.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`True`):
    > The scene features a tube house, a grand neoclassic theatre facade, and a colonial building. There is also a carved stone dragon staircase and a yellow classical mansion estate. Nearby, there is an exhibition of historic red banners and a yellow colonial museum facade. Additionally, a multi-tiered stone pavilion and a historic brick flag tower can be seen, though their exact locations are less certain.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`12.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`True`):
    > The scene features a tube house, a grand neoclassic theatre facade, and a colonial building. There is also a carved stone dragon staircase and a yellow classical mansion estate. Nearby, there is an exhibition of historic red banners and a yellow colonial museum facade. Additionally, a multi-tiered stone pavilion and a historic brick flag tower can be seen, though their exact locations are less clear.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`12.5%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a tube house, a grand neoclassic theatre facade, and a colonial building. There is also a carved stone dragon staircase and a yellow classical mansion estate. Nearby, there is an exhibition of historic red banners and a yellow colonial museum facade. Additionally, a multi-tiered stone pavilion and a historic brick flag tower can be seen, though their exact locations are unclear.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 27 — `hcmc_0085` [BucketD_CorrectID_LOW]
- **GT Landmark:** `Ba Thien Hau Temple` | **YouTube ID:** `ebNzfxPO_Hg` | **Tier:** `10s` | **p_onsite:** `0.9999`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Chua Ba Thien Hau/segment/Chùa Bà Thiên Hậu - Chợ Lớn [ebNzfxPO_Hg]_scene_002.mp4`
- **Z4 5-way:** `Ba Thien Hau Temple` (`mu_L=0.4991`, `LOW`) | **Z4 35-way:** `Ba Thien Hau Temple` (`mu_L=0.1808`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows the exterior of a traditional Vietnamese temple with ornate, tiled roofing and decorative carvings above the entrance. Red lanterns hang from poles on either side of the gate, and a red banner with white text is displayed above the main doorway, with visible signage reading “Chùa Bà Thiên Hậu” and “Chùa Lớn” below it.
- **Indep-GT Gold Facts:**
  - `(Ba Thien Hau Temple, has_part, roof)` | mu_F=0.99049 | vis_R=1.0 (qwen=1.0, clip=0.9432)
  - `(Ba Thien Hau Temple, has_part, main structure)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.989, clip=0.9531)
  - `(Ba Thien Hau Temple, located_in, Ho Chi Minh City)` | mu_F=0.98582 | vis_R=None (qwen=None, clip=None)
  - `(Ba Thien Hau Temple, located_in, Cholon)` | mu_F=0.75061 | vis_R=None (qwen=None, clip=None)
  - `(Ba Thien Hau Temple, dedicated_to, Mazu)` | mu_F=0.67792 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `None` (abstained at LOW mu_L < 0.50)
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`40.0%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > Ba Thien Hau Temple, located in Cholon, Ho Chi Minh City, features an upper wall and a traditional courtyard sanctuary. Inside, one can see a multi-tiered stone pavilion and a traditional three-door gate, while the sacred inner shrine altar and imperial palace are also part of the complex. Adjacent to the temple, there is a multi-tiered red Buddhist stupa, adding to the rich cultural tapestry of this historic site.
  - **B3_Prompt_Only_Certainty_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features several visually distinct elements, including a temple den, a pagoda, a sacred inner shrine altar, a traditional temple courtyard sanctuary, a multi-tiered stone pavilion, an ancient stone citadel gate, a traditional temple three-door gate, a conical palm hat artistic display, an imperial palace, and a multi-tiered red Buddhist stupa. However, the exact landmark identity remains uncertain.
  - **B4_Ours_No_VCGAT_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a temple den, a pagoda, a sacred inner shrine altar, and a traditional temple courtyard sanctuary. There is also a multi-tiered stone pavilion and an ancient stone citadel gate. Additionally, there are conical palm hats on display, though their exact purpose is unclear. The area appears to be part of a larger traditional temple complex.
  - **Proposed_VCGAT_E2E_5way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a temple den, a pagoda, a sacred inner shrine altar, and a traditional temple courtyard sanctuary. There is also a multi-tiered stone pavilion and an ancient stone citadel gate. Additionally, there are conical palm hats on display, though their exact purpose is unclear. The area appears to be part of a larger traditional temple complex.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a temple den, a pagoda, a sacred inner shrine altar, and a traditional temple courtyard sanctuary. There is also a multi-tiered stone pavilion and an ancient stone citadel gate. Additionally, there are conical palm hats on display, though their exact purpose is unclear.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B4 -> Qwen=`Tie`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 28 — `hcmc_0934` [BucketD_CorrectID_MEDIUM]
- **GT Landmark:** `Saigon Central Post Office` | **YouTube ID:** `0drdaM-VYsc` | **Tier:** `20s` | **p_onsite:** `1.0`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Buu dien trung tam Sai Gon/segment/Bưu điện trung tâm Sài Gòn [0drdaM-VYsc]_scene_001.mp4`
- **Z4 5-way:** `Saigon Central Post Office` (`mu_L=0.5112`, `MEDIUM`) | **Z4 35-way:** `Saigon Central Post Office` (`mu_L=0.2444`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows the exterior of a large, ornate building with a yellowish-beige facade, arched windows, and decorative stonework, featuring a prominent central entrance with a metal gate and a clock above it. People are visible walking in front of the building, and modern structures can be seen in the background under a partly cloudy sky.
- **Indep-GT Gold Facts:**
  - `(Saigon Central Post Office, has_part, facade)` | mu_F=0.99991 | vis_R=1.0 (qwen=0.9998, clip=0.999)
  - `(Saigon Central Post Office, has_part, exterior)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9998, clip=0.9972)
  - `(Saigon Central Post Office, has_part, grand entrance)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9996, clip=0.9927)
  - `(Saigon Central Post Office, located_in, Ho Chi Minh City)` | mu_F=0.94963 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Central Post Office, located_in, District 1)` | mu_F=0.83925 | vis_R=None (qwen=None, clip=None)
  - `(Saigon Central Post Office, designed_by, Jules Bourard)` | mu_F=0.74855 | vis_R=None (qwen=None, clip=None)
  - `(plaques, commemorates, scientists)` | mu_F=0.45312 | vis_R=None (qwen=None, clip=None)
  - `(plaques, commemorates, engineers)` | mu_F=0.44976 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Saigon Central Post Office, has_part, plaques)` | c_f=0.4614 | mu_F=0.9025 | vcgat=11.354
  - `(Saigon Central Post Office, has_part, interior walls)` | c_f=0.4614 | mu_F=0.9025 | vcgat=11.2841
  - `(Saigon Central Post Office, has_part, design)` | c_f=0.4614 | mu_F=0.9025 | vcgat=11.1832
  - `(historical maps, dated_to, 1892)` | c_f=0.1674 | mu_F=0.32752 | vcgat=10.6749
  - `(Saigon Central Post Office, built_by, French architects)` | c_f=0.1428 | mu_F=0.27944 | vcgat=10.6368
  - `(Saigon Central Post Office, built_by, engineer)` | c_f=0.1074 | mu_F=0.21 | vcgat=10.4011
  - `(plaques, commemorates, engineers)` | c_f=0.2299 | mu_F=0.44976 | vcgat=11.3621
  - `(interior walls, architectural_style, Indo-Saracenic architecture)` | c_f=0.1341 | mu_F=0.26224 | vcgat=11.3595
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`62.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, a prominent landmark in Ho Chi Minh City's District 1, features an ornate facade adorned with arched windows and a clock tower. Historical maps from 1892 depict its presence, while plaques commemorate the contributions of scientists and engineers. Amidst the colonial architecture, the post office stands out with its neoclassical elements, including a grand facade and a multi-tiered stone pavilion. Tourists gather around this historic site, appreciating its blend of architectural styles and historical significance.
  - **B3_Prompt_Only_Certainty_5way** (regime=`MEDIUM`, K-Rec=`12.5%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene likely shows a structure that appears to be the Saigon Central Post Office, featuring plaques and interior walls that possibly exhibit Indo-Saracenic architecture. If this is indeed the Saigon Central Post Office, it was likely built by French architects and an engineer, and its interior walls may feature elements of Indo-Saracenic design. The building also includes historical plaques commemorating engineers, adding to its historical significance.
  - **B4_Ours_No_VCGAT_5way** (regime=`MEDIUM`, K-Rec=`37.5%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Saigon Central Post Office, likely showing an arched window and a clock tower, stands prominently in District 1 of Ho Chi Minh City. Its facade, though not clearly visible, appears to be part of a larger colonial building. Nearby, there is a yellow classical mansion estate and a multi-tiered stone pavilion, suggesting a rich historical context. Tourists can be seen exploring the area, though the exact identity of the landmark is uncertain.
  - **Proposed_VCGAT_E2E_5way** (regime=`MEDIUM`, K-Rec=`25.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Saigon Central Post Office likely shows elements of Indo-Saracenic architecture, including its design and interior walls. Plaques commemorating engineers are present, though their exact placement is uncertain. The building itself appears to be part of a larger colonial structure, possibly a yellow classical mansion estate or a yellow colonial museum facade.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`12.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`True`):
    > The scene features a yellow classical mansion estate, a colonial building, and a yellow colonial museum facade. There is also a multi-tiered stone pavilion and a grand neoclassic theatre facade. Tourists can be seen near these structures.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`Tie` | vs B4 -> Qwen=`B4_Ours_No_VCGAT_5way`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 29 — `hcmc_0537` [BucketD_CorrectID_MEDIUM]
- **GT Landmark:** `Independence Palace` | **YouTube ID:** `1vK0TyASAIA` | **Tier:** `30s` | **p_onsite:** `0.9959`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Dinh Doc Lap/segment/Du lịch Việt Nam： Dinh Độc Lập – di tích lịch sử quốc gia đặc biệ [1vK0TyASAIA]_scene_002.mp4`
- **Z4 5-way:** `Independence Palace` (`mu_L=0.5359`, `MEDIUM`) | **Z4 35-way:** `Yen So park` (`mu_L=0.212`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** A large, modernist-style building with a flat roof and multiple flagpoles displaying red flags is visible, with a sign above reading “INDEPENDENCE PALACE” and “PALAIS DE L’INDÉPENDANCE.” In front of the building, a wide lawn features a central fountain surrounded by trees, with a paved pathway leading toward the entrance.
- **Indep-GT Gold Facts:**
  - `(Independence Palace, has_part, beautiful garden)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9933, clip=0.9737)
  - `(Independence Palace, has_part, white facade)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9707, clip=0.9605)
  - `(Independence Palace, has_part, ground plan)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.6792, clip=0.9984)
  - `(Independence Palace, has_part, facade)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.6792, clip=0.9588)
  - `(Independence Palace, located_in, Ho Chi Minh City)` | mu_F=0.98933 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, built_by, French colonial authorities)` | mu_F=0.93663 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, designed_by, Ngô Viết Thụ)` | mu_F=0.88484 | vis_R=None (qwen=None, clip=None)
  - `(Independence Palace, located_in, District 1)` | mu_F=0.83493 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Independence Palace, part_of, Republic of Vietnam)` | c_f=0.1255 | mu_F=0.23416 | vcgat=10.6757
  - `(Independence Palace, has_part, Independence Hall)` | c_f=0.5308 | mu_F=0.99049 | vcgat=9.2818
  - `(Independence Palace, has_part, hall)` | c_f=0.4836 | mu_F=0.9025 | vcgat=9.0254
  - `(Independence Palace, part_of, city centre)` | c_f=0.2248 | mu_F=0.41947 | vcgat=8.9996
  - `(Independence Palace, commissioned_by, Republic of Vietnam)` | c_f=0.1679 | mu_F=0.31336 | vcgat=10.7036
  - `(Independence Palace, commissioned_by, Ngô Đình Diệm)` | c_f=0.3764 | mu_F=0.70238 | vcgat=10.6918
  - `(Independence Palace, built_by, Ngô Đình Diệm)` | c_f=0.2961 | mu_F=0.55247 | vcgat=10.6894
  - `(Independence Palace, built_in, 1962)` | c_f=0.2494 | mu_F=0.46537 | vcgat=10.6112
  - `(Independence Palace, part_of, Ho Chi Minh City)` | c_f=0.1112 | mu_F=0.20744 | vcgat=8.4374
  - `(Independence Palace, architectural_style, Baroque Revival)` | c_f=0.1589 | mu_F=0.29648 | vcgat=8.3527
  - `(Independence Palace, surrounded_by, Ho Chi Minh City)` | c_f=0.2702 | mu_F=0.50416 | vcgat=8.3161
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Independence Palace stands prominently in the heart of Ho Chi Minh City's bustling District 1, its imposing facade reflecting the architectural style influenced by French colonial authorities. Central to the palace grounds is a towering flagpole, flanked by sun-shading louvers designed to protect against the scorching sun. Adjacent to this is a rooftop pavilion offering panoramic views of the city. The expansive manicured green lawn stretches out before the palace, providing a serene backdrop for state banquets and presidential receptions held within the grand reception halls. Nearby, traditional silk tunic exhibitions and historic red banners exhibitions showcase the rich cultural heritage of the region. While the palace's interior features a museum, it also houses an inner sacred shrine altar, adding to its significance as a symbol of national pride.
  - **B3_Prompt_Only_Certainty_5way** (regime=`MEDIUM`, K-Rec=`12.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > The visual elements observed likely show Independence Palace, which appears to be part of the city centre of Ho Chi Minh City. The structure includes Independence Hall, possibly featuring Baroque Revival architecture. It is likely that the palace was commissioned by the Republic of Vietnam under the leadership of Ngô Đình Diệm, who also oversaw its construction in 1962. Surrounding the palace, one can see an expansive manicured green lawn and other historical structures such as a historic brick flag tower and a temple den. If this is indeed Independence Palace, it stands as a significant architectural piece reflecting the period of Vietnamese history under the Republic of Vietnam.
  - **B4_Ours_No_VCGAT_5way** (regime=`MEDIUM`, K-Rec=`50.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Independence Palace, likely showing its central flagpole and ground plan, stands prominently in Ho Chi Minh City's District 1. It features sun-shading louvers and a rooftop pavilion, though these elements may not be fully visible. The palace, built by the French colonial authorities, includes state banquet areas and presidential reception halls. Nearby, there are expansive manicured lawns and an imperial palace, though their exact identification is uncertain.
  - **Proposed_VCGAT_E2E_5way** (regime=`MEDIUM`, K-Rec=`12.5%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Independence Palace, if this is indeed Independence Palace, appears to be located in the heart of Ho Chi Minh City's city centre. It features a Baroque Revival architectural style and is surrounded by expansive manicured lawns. The palace itself includes an Independence Hall, though its exact identity is uncertain. Nearby, there are other notable structures such as a traditional silk tunic exhibition and a water spray show, suggesting a rich cultural heritage.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features an expansive manicured green lawn, an imperial palace, a traditional silk tunic exhibition, a water spray show, a yellow classical mansion estate, a historic brick flag tower, a sacred inner shrine altar, historic red banners on display, and a temple den. However, the exact landmark identity remains uncertain.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`B2_Crisp_KG_RAG_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`B3_Prompt_Only_Certainty_5way` | vs B4 -> Qwen=`B4_Ours_No_VCGAT_5way`, Llama=`B4_Ours_No_VCGAT_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___

## Case 30 — `hcmc_0299` [BucketD_CorrectID_MEDIUM]
- **GT Landmark:** `Ben Thanh Market` | **YouTube ID:** `AGhll-2UgwQ` | **Tier:** `30s` | **p_onsite:** `0.9991`
- **Video Segment (server):** `/home/nguyendinhquy/TransMTL_OntoKG-main/data/VIDEO_TEST_HCMC/Cho Ben Thanh/segment/🛍️ Từ Chợ Bến Thành đến Takashimaya ｜ Khám phá Sài Gòn từ truyền [AGhll-2UgwQ]_scene_007.mp4`
- **Z4 5-way:** `Ben Thanh Market` (`mu_L=0.6117`, `MEDIUM`) | **Z4 35-way:** `Hanoi Old Quarter` (`mu_L=0.4734`, `LOW`)
- **Blind VLM Visual Desc (`Qwen3-VL-8B`):** The video shows a bustling covered market with arched ceilings and hanging signs in Vietnamese, featuring vendors and shoppers moving through narrow aisles lined with stalls selling various goods. Brightly colored merchandise and signage are visible, with some stalls displaying flags and others showcasing items like clothing and souvenirs under artificial lighting.
- **Indep-GT Gold Facts:**
  - `(Ben Thanh Market, has_part, interior)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9959, clip=0.9904)
  - `(Ben Thanh Market, has_part, interior hall)` | mu_F=0.99907 | vis_R=1.0 (qwen=0.9948, clip=0.9469)
  - `(Ben Thanh Market, has_part, roofs)` | mu_F=0.99049 | vis_R=1.0 (qwen=0.9994, clip=0.9361)
  - `(Ben Thanh Market, has_part, market hall)` | mu_F=0.9025 | vis_R=1.0 (qwen=0.9968, clip=0.9077)
  - `(Ben Thanh Market, located_in, Ho Chi Minh City)` | mu_F=0.99979 | vis_R=None (qwen=None, clip=None)
  - `(Ben Thanh Market, built_in, early 1900s)` | mu_F=0.83871 | vis_R=None (qwen=None, clip=None)
  - `(Ben Thanh Market, designed_by, Valentin de La Croix)` | mu_F=0.81427 | vis_R=None (qwen=None, clip=None)
  - `(Ben Thanh Market, located_in, Vietnam)` | mu_F=0.76258 | vis_R=None (qwen=None, clip=None)
- **Proposed_5way Input `GIVEN FACTS`:**
  - `(Ben Thanh Market, part_of, city)` | c_f=0.2742 | mu_F=0.44828 | vcgat=10.6755
  - `(Ben Thanh Market, part_of, the city)` | c_f=0.1706 | mu_F=0.27888 | vcgat=10.6313
  - `(southern clock tower, part_of, new market hall)` | c_f=0.1773 | mu_F=0.28992 | vcgat=10.3106
  - `(Ben Thanh Market, architectural_style, French-style)` | c_f=0.446 | mu_F=0.72904 | vcgat=10.1394
  - `(Ben Thanh Market, founded_in, 1914)` | c_f=0.4225 | mu_F=0.69069 | vcgat=10.9287
  - `(Ben Thanh Market, built_by, Valentin de La Croix)` | c_f=0.3612 | mu_F=0.59049 | vcgat=10.9045
  - `(Ben Thanh Market, built_by, architect)` | c_f=0.1229 | mu_F=0.20088 | vcgat=10.7748
  - `(Ben Thanh Market, located_in, the city)` | c_f=0.3295 | mu_F=0.53874 | vcgat=10.6189
  - `(Ben Thanh Market, architectural_style, French)` | c_f=0.1553 | mu_F=0.25392 | vcgat=10.0655
  - `(Ben Thanh Market, architectural_style, Vietnamese)` | c_f=0.1886 | mu_F=0.3084 | vcgat=9.7684
  - `(Ben Thanh Market, part_of, Ho Chi Minh City)` | c_f=0.3702 | mu_F=0.60512 | vcgat=9.6326
- **Generated Captions & Indep-GT Audit:**
  - **B2_Crisp_KG_RAG_5way** (regime=`CRISP`, K-Rec=`75.0%`, FactP=`0.0%`, VSP_VLM=`85.7%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > Ben Thanh Market, a historic landmark in Ho Chi Minh City, Vietnam, stands as a testament to the city's rich cultural heritage. Built in 1914, it features a southern clock tower, an ornate interior hall with domed roofs, and columns that adorn its facade. The market's main hall and arches provide a vibrant backdrop for the bustling evening pedestrian market, where tourists can observe the colorful silk lanterns and ethnic minority women's costumes. Alongside the traditional silk tunic exhibition, the market also showcases street food stalls and woven silk fabric bolts, creating a lively atmosphere that reflects the local community's traditions and crafts.
  - **B3_Prompt_Only_Certainty_5way** (regime=`MEDIUM`, K-Rec=`37.5%`, FactP=`0.0%`, VSP_VLM=`100.0%`, OAR=`True`, TrueErr=`False`, LowStyleOnly=`False`):
    > In the video, Ben Thanh Market likely shows signs of its French-style architecture, which appears to be a prominent feature of the bustling evening pedestrian market. The market, possibly part of Ho Chi Minh City, features both Vietnamese and French elements, as evidenced by the colorful silk lanterns and traditional silk tunic exhibitions. If this is indeed Ben Thanh Market, it stands as a testament to the city's rich history, founded in 1914 and constructed by architect Valentin de La Croix. Alongside these architectural elements, the scene includes various vendors displaying woven silk fabric bolts and offering street food from sidewalk stalls. The presence of a tourist group further suggests that Ben Thanh Market remains an active and popular part of the city's cultural landscape.
  - **B4_Ours_No_VCGAT_5way** (regime=`MEDIUM`, K-Rec=`37.5%`, FactP=`100.0%`, VSP_VLM=`80.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The Ben Thanh Market, if this is indeed Ben Thanh Market, features a southern clock tower, an interior hall, and a domed roof. Ornate columns support the structure, which also includes a main hall and a facade. The market likely showcases woven silk fabric bolts displays and a bustling evening pedestrian market with street food stalls. Tourists can observe ethnic minority women wearing costumes and conical palm hats, while colorful silk lanterns hang overhead in a canopy walkway.
  - **Proposed_VCGAT_E2E_5way** (regime=`MEDIUM`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > Ben Thanh Market, likely showing French-style architecture, can be seen in the scene. The market features a bustling evening pedestrian market with woven silk fabric bolts displayed on the ground. Tourist groups can be observed, along with street food stalls and colorful silk lanterns hanging overhead. The market also includes tube houses and ethnic minority women's costumes, creating a vibrant atmosphere.
  - **Proposed_VCGAT_E2E_35way** (regime=`LOW`, K-Rec=`0.0%`, FactP=`100.0%`, VSP_VLM=`100.0%`, OAR=`False`, TrueErr=`False`, LowStyleOnly=`False`):
    > The scene features a bustling evening pedestrian market with colorful silk lanterns hanging overhead. Tourists can be seen exploring the area, while ethnic minority women wear traditional costumes. Sidewalk street food stalls line the streets, and woven silk fabric bolts are displayed prominently. The atmosphere is lively, with a mix of modern and traditional elements.
- **Dual Judges (overall_winner):** vs B2 -> Qwen=`B2_Crisp_KG_RAG_5way`, Llama=`Proposed_VCGAT_E2E_5way` | vs B3 -> Qwen=`B3_Prompt_Only_Certainty_5way`, Llama=`B3_Prompt_Only_Certainty_5way` | vs B4 -> Qwen=`Proposed_VCGAT_E2E_5way`, Llama=`Proposed_VCGAT_E2E_5way`
- **Human Audit Sign-Off:** `[ ] Scene accurate  [ ] Proposed faithful  [ ] True factual error in Proposed? (Y/N)  [ ] Notes:` ___
