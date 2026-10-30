# Nutrify AI - Data & Image Source Licenses

Food-101: not used.

## Image License Policy Gate
- **Accepted Image Licenses**:
  - CC0 1.0 Universal (Public Domain Dedication)
  - Public Domain (PD / US Government Work). Public domain status is jurisdiction-dependent. Accept a PD image only when its Commons licence tag is explicitly PD or CC0, and record the tag.
  - Creative Commons Attribution (CC BY 2.0, CC BY 2.5, CC BY 3.0, CC BY 4.0)
- **Strictly Rejected Image Licenses**:
  - CC BY-SA (any version)
  - CC BY-NC (any version)
  - CC BY-ND (any version)
  - GFDL (GNU Free Documentation License)
  - Unknown license / missing extmetadata

---

## Data & Media Sources

### 1. USDA FoodData Central (Foundation Foods & SR Legacy)
- **Status**: ACCEPTED
- **Official URL**: https://fdc.nal.usda.gov/
- **Dataset Download Page**: https://fdc.nal.usda.gov/download-datasets.html
- **Licence URL / Reference**: https://fdc.nal.usda.gov/faq.html#q21
- **Policy & Legal Note**:
  USDA FoodData Central data is treated as a U.S. Government work (17 U.S.C. 105). The FAQ permits commercial software use. Legal basis to be confirmed by counsel before commercial launch.
- **Fetched FAQ Text (https://fdc.nal.usda.gov/faq.html#q21)**:
  "USDA has been the authoritative source of food composition data for more than a century, and the primary source of publicly available data on the nutrients and other components found in foods. Users of these data include academic research institutions, federal agencies, nutrition and health professionals, software developers, foreign governments and organizations, the food industry, restaurants, and others. By combining a robust mix of food, nutrient, and other food component data in one system, FoodData Central can strengthen current uses of the data and facilitate new uses. The data in FoodData Central can be used in a variety of ways, such as ... Developing commercial nutrient analysis software and applications for private sector purposes."
- **What it provides**: Ground-truth nutrient composition datasets (Foundation Foods CSV, SR Legacy CSV) providing calories, macronutrients, micronutrients, and standard serving portion weights.
- **What is forbidden**: Implying USDA endorsement of commercial software/products; misrepresenting Government data.
- **Date Verified**: 2026-10-30

### 2. Wikimedia Commons
- **Status**: ACCEPTED (subject to image license filter gate)
- **Official URL**: https://commons.wikimedia.org/
- **API Endpoint**: https://commons.wikimedia.org/w/api.php
- **Licence URL**: https://commons.wikimedia.org/wiki/Commons:Licensing
- **Fetched Licensing Text (https://commons.wikimedia.org/wiki/Commons:Licensing)**:
  "All copyrighted material on Commons (not in the public domain ) must be licensed under a free license that specifically and irrevocably allows anyone to use the material for any purpose; simply writing that " the material may be used freely by anyone " or similar isn't sufficient. In particular, the license must meet the following conditions: Republication and distribution must be allowed. Publication of derivative work must be allowed. Commercial use of the work must be allowed. The license must be perpetual (non-expiring) and non-revocable."
  "The following well-known licenses are preferred for materials on Commons: Creative Commons Attribution/ShareAlike/Zero licenses"
- **What it provides**: Freely usable photographs of standard foods and culinary ingredients.
- **What is forbidden**:
  - Using images under CC BY-SA, CC BY-NC, CC BY-ND, GFDL, or unverified licenses.
  - Sending automated requests without identifying User-Agent (`NutrifyAI/1.0 (akshkanani01@gmail.com)`).
  - Exceeding the rate limit of 1 request per second or omitting `maxlag=5`.
- **Attribution & Usage Notes**:
  - CC-BY images require attribution wherever the image is displayed. Reference images are not displayed in the MVP. If they are displayed later, attribution must be shown. Obligations for a model trained on CC-BY images: to be reviewed by counsel.
- **Date Verified**: 2026-10-30

### 3. Open Food Facts
- **Status**: NOT USED (decision A)
- **Reason**: The master CSV is published in a public repository, and ODbL share-alike plus CC-BY-SA product images would apply.
- **Official URL**: https://world.openfoodfacts.org/
- **Dataset Download Page**: https://world.openfoodfacts.org/data
- **Licence URL**: https://opendatacommons.org/licenses/odbl/1.0/
- **Date Verified**: 2026-10-30

### 4. IFCT / ICMR-NIN (Indian Food Composition Tables)
- **Status**: UNVERIFIED (EXCLUDED)
- **Official URL**: http://www.ifct2017.com/
- **Licence URL**: N/A (no public bulk download verified in Step 2)
- **What it provides**: Regional nutritional composition for Indian recipes and foods.
- **Reason for exclusion**: Official direct bulk CSV was not verified in Step 2. Values must come from a user-supplied local CSV.
- **Date Verified**: 2026-10-30
