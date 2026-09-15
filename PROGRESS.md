# PROGRESS.md — وضعیت پروژه

آخرین به‌روزرسانی: 2026-09-13 (دور دوم) — **اصلاحات جلسه استاد/داور اعمال شد**:
عنوان کوتاه، چکیده ۱۸۱ کلمه‌ای، ساختار ۶ فصلی (XAI → Discussion،
Corruption Framework → زیربخش ۴.۱، Conclusion → زیربخش پایانی Discussion)،
نتیجه‌گیری ۱۹۰ کلمه‌ای، ایمیل‌های جدید (imanabedini.ai@gmail.com؛
kpeyvandi@semnan.ac.ir) و affiliation دانشگاه سمنان (فقط [Department] مانده).

## ۱.۸) دور اصلاحات جلسه (2026-09-13)

| مورد جلسه | اقدام |
|---|---|
| عنوان طولانی | → «PD-RobustBench: A Corruption-Robustness and Explainability Framework for Parkinson's Disease Detection from Hand-Drawn Images» (تک‌خطی) |
| چکیده ~۳۵۰ کلمه | → ۱۸۱ کلمه (بازه ۱۵۰–۲۰۰)؛ انتهای تکراری حذف شد |
| ۹ فصل | → ۶ فصل: 1 Intro, 2 Related, 3 Data&Models, 4 Evaluation Metrics and Robustness Score (4.1 Corruption Framework, 4.2 Evaluation metrics, 4.3 Relative degradation, 4.4 XAI stability term, 4.5 PRS), 5 Results, 6 Discussion (6.1 XAI Analysis شامل stability/qualitative/sanity، بعد practical implications، score caveat، Limitations، Future work، 6.x Conclusion) |
| Conclusion ~۱۰۵ کلمه | → ۱۹۰ کلمه با سه نتیجه‌گیری صریح |
| ایمیل‌ها | Iman: imanabedini.ai@gmail.com؛ دکتر پیوندی: kpeyvandi@semnan.ac.ir (نویسنده مسئول)؛ affiliation: Semnan University |
| نگرش AI-نویسی | چکیده/نتیجه‌گیری از نو و با ساختار متفاوت نوشته شد؛ چند عبارت قالبی حذف شد. نکته: بازنویسی کامل «انسانی» باید توسط دانشجویان با روخوانی انجام شود؛ در صورت تصمیم به ارسال فارسی، نسخه فارسی کامل قابل تولید است. |
| بسته Overleaf | paper_submission.zip بازسازی شد (main + supplementary + شکل‌ها + جدول‌ها) |

اسکریپت ساختار: `docs/paper/restructure_paper.py` (قابل بازبینی/حذف).

## ۱.۷) دور توسعه (2026-09-13) — «اگر زمان داشتید»های review

| مورد | اقدام |
|---|---|
| ۳ seed | کل پایپ‌لاین (آموزش دومرحله‌ای → robustness → XAI) برای seedهای 42/43/44 اجرا شد. خروجی‌ها: `results/parts/s42|s43|s44/`، `robustness_results_s43/s44.csv`، `robustness_scores_s43/s44.csv`، `seed_aggregate.csv` (میانگین±انحراف دقت/AUC/PRS) + جدول tab:seeds و پاراگراف Seed variability در مقاله. یافته: بازه PRS بین seedها 0.3–10.5 واحد؛ رتبه‌بندی روی PDRAW بین seedها جابه‌جا می‌شود (ρ جفتی −0.8 تا +0.8) و روی HandPD نسبتاً پایدار است (0.4–1.0). بهترین میانگین PRS: EffNet روی هر دو PDRAW، VGG16 و ResNet50 هر کدام روی یک نوع HandPD. |
| مدل چهارم | MobileNetV3-Large (3.0M پارامتر، 1.81M trainable) از طریق همان interface سه‌نقطه‌ای اضافه شد → ۱۶ پیکربندی. sanity check گراد-کم هم برایش اجرا شد (ρ=0.07). |
| perspective + shadow | دو خانواده جدید × ۵ سطح (پیکربندی در config.py؛ اعمال قطعی) → مجموعاً ۸ خانواده، ۴۱ شرط. یافته جدید: shadow دومین خانواده مخرب (3–9٪) بعد از rotation (10–22٪)؛ perspective ملایم (تا 6.4٪)؛ در XAI هم perspective دومین بی‌ثبات‌کننده (ρ̄=0.66). |
| بازتولید کامل | همه جداول/ردیف‌های LaTeX، شکل‌ها (گرید ۲×۴ برای ۸ خانواده)، CIهای بوت‌استرپ، حساسیت وزن‌ها (اکنون: وزن برابر ρ=1.0، performance-only ρ=0.95) و sanity با ۴ مدل بازتولید شدند. |
| مقاله | همه اعداد متن/چکیده/بحث/نتیجه با نتایج جدید به‌روز؛ دو placeholder عمدی باقی است ([Department]… و [REPOSITORY-URL]). |
| بسته Overleaf | `docs/paper/paper_submission.zip` بازسازی شد (main + supplementary + ۹ شکل + ۷ فایل ردیف جدول). |

## ۱) خلاصه وضعیت

| فاز | وضعیت |
|-----|-------|
| ۱.۱ جدول مدل‌ها/دیتاست‌ها (ادبیات) | ✅ `docs/task1_1_models_datasets.md` + `.csv` |
| ۱.۲ مرور کارهای robustness | ✅ `docs/task1_2_related_work_summary.md` |
| ۲ دیتاست‌ها و مدل‌ها | ✅ PDRAW + HandPD؛ ۴ backbone (ResNet50/VGG16/EffNet-B0/MobileNetV3-Large) |
| ۳ خراب‌سازی‌ها | ✅ ۸ خانواده × ۵ سطح (شامل shadow و perspective) |
| ۴ ارزیابی | ✅ هر seed: ۶۵۶ ردیف robustness (16×41) + 45,264 پیش‌بینی per-image |
| ۵ XAI | ✅ هر seed: 6,400 مقایسه CAM + sanity check ۴ مدلی |
| ۶ PRS | ✅ per-seed + aggregate سه seed + حساسیت وزن‌ها |
| ۷ مقاله/README/PROGRESS | ✅ به‌روز؛ بسته Overleaf آماده |

## ۱.۴) دور اصلاحات پیش از داوری (2026-09-11)

| مورد review | اقدام |
|---|---|
| placeholderها | نویسندگان: Ramtin Torik, Iman Abedini, Kimia Peyvandi (corresponding, kimiapeyvandi@gmail.com). فقط `[Department], [University]` و `[REPOSITORY-URL]` عمداً برای کاربر مانده |
| تناقض threshold | هر دو بخش روش/معیارها یکدست شدند: threshold منتخب val برای همه metricهای threshold-based؛ جدول S1 مقادیر |
| تطبیق فرمول/جدول PRS | robustness_score.py با clip بازنویسی و اجرا شد؛ PRSها با جدول‌ها مو در میان (مثلاً 92.2/86.9/81.8 در meander) |
| CI / حجم test | `src/supplementary_stats.py` → bootstrap 95% (2000 resample): عرض CI دقت ±5.6–16.7 واحد؛ پاراگراف Uncertainty در Results + جدول S2 |
| ادعای «اولین» | → "to the best of our knowledge, among the first systematic..." |
| «realistic» | → "synthetic, acquisition-inspired" در چکیده/مقدمه/محدودیت‌ها |
| single-seed | محدودیت (v) تقویت + ادعاهای "highest observed ... under the evaluated split and seed" |
| Data availability | فهرست کامل مصنوعات + URL placeholder |
| حساسیت وزن‌های PRS | `src/prs_sensitivity.py` → سه وزن‌دهی، رتبه‌بندی یکسان (ρ=1.0)، جدول S3 |
| پارامترها | ResNet50 23.6M/15.1M، VGG16 14.7M/7.11M، EffNet 4.1M/3.24M در بخش Models |
| جدول split/سوژه‌ها | جدول tab:split + محدودیت صریح نبودن شناسه سوژه در PDRAW |
| sanity check Grad-CAM | `src/gradcam_sanity.py` → ρ( trained vs untrained ): 0.23/−0.15/0.04 + زیربخش 5.3 «Stability is not validity» |
| interface مدل جدید | زیربخش 3.3 Extensibility با دستور نمونه |
| supplementary | `docs/paper/submission/supplementary.tex` (S1 thresholds، S2 CI، S3 sensitivity، S4 منحنی بقیه dtypes، S5 sanity) |
| بسته Overleaf | `docs/paper/paper_submission.zip` (خودکفا: main + supplementary + ۹ شکل + ۶ جدول) |

## ۱.۵) یافته‌های کلیدی (برای مرور سریع)

- **PRS (0-100، بالاتر = مقاوم‌تر):** بازه 81.8–96.4؛ EffNet-B0 در ۳ از ۴ نوع داده بهترین
  (PDRAW-spiral 94.5، PDRAW-wave 96.4، HandPD-spiral 87.1) و در HandPD-meander بدترین (81.8،
  آنجا ResNet50 با 92.2 بهترین است).
- **Rotation مخرب‌ترین خراب‌سازی است:** افت balanced accuracy 10–22٪ و پایداری CAM
  (Spearman) تا 0.27 در شدت 5.
- **Contrast تقریباً بی‌اثر** (−1 تا +2.6٪؛ CAM ρ=0.91). JPEG پیش‌بینی‌ها را تقریباً حفظ
  می‌کند ولی توضیحات را بازنویسی می‌کند (ρ: 0.96→0.62).
- پایه‌های تمیز: HandPD دقت 0.83–0.90 (AUC تا 0.98)؛ PDRAW دقت 0.73–0.90 (AUC 0.82–0.94).

## ۱.۶) تست‌های انجام‌شده ✅

1. corruption smoke test (۶×۵، قطعیت RNG، گالری بصری) — پاس
2. forward-pass هر ۳ backbone (ابعاد ویژگی 2048/512/1280) — پاس
3. Grad-CAM دستی روی هر ۳ مدل (پس از رفع ۲ باگ: requires_grad ورودی؛ لایه هدف VGG=features[30]) — پاس
4. clean inference با threshold منتخب val — پاس (نمونه: pdraw_spiral/ResNet50 AUC=0.924)
5. integrity check پس از merge: هر ۱۲ پیکربندی دقیقاً ۳۱ شرط دارد — پاس
6. جداول LaTeX از CSV تولید شدند (بدون ورود دستی اعداد) — پاس

## ۲) محیط

- Windows 11، Python 3.10.11 (فقط Intel Arc GPU بدون CUDA → همه‌چیز CPU)
- نصب‌شده: torch 2.14.0+cpu, torchvision 0.29.0+cpu, cv2 5.0.0, numpy 2.2.6,
  pandas, scikit-learn 1.7.2, scikit-image, scipy, matplotlib, seaborn, kagglehub, Pillow
- وزن‌های ImageNet در `C:\Users\ramti\.cache\torch\hub\checkpoints\`
  (resnet50-11ad3fa6 ✓، vgg16-397923af ✓، efficientnet_b0_rwightman-7f5810bc ✓ — هش تأیید شده)

## ۳) دیتاست‌ها (آماده در `data/raw/`)

| کلید | train | test | منبع |
|------|-------|------|------|
| pdraw_spiral / pdraw_wave | 72 (36/36) | 30 (15/15) | Kaggle kmader/parkinsons-drawings (کش kagglehub) |
| handpd_spiral / handpd_meander | 260 (208/52) | 108 (88/20) | UNESP HandPD — تقسیم subject-wise 70/30 با seed=42 |

- کلاس‌ها: healthy=0، parkinson=1. HandPD عمداً ~4:1 نامتوازن است (بالتوجه: balanced metrics).
- تصاویر 256×256 PNG (PDRAW) و ~700px JPG (HandPD)؛ همه به 224×224 با نرمال‌سازی ImageNet.

## ۴) پروتکل مدل‌ها (تصویب‌شده توسط کاربر: «همین فعلی»)

- **مرحله ۱:** backbone کاملاً فروز (وزن ImageNet) + head سبک یکسان
  (GAP→Dropout0.3→Linear64→ReLU→Linear1) روی ویژگی‌های کش‌شده،
  Adam lr=1e-3، حداکثر ۱۰۰ epoch، early-stop روی val AUC (تقسیم val = 20٪ stratified از train، seed=42).
- **مرحله ۲:** فقط آخرین بلوک backbone (ResNet50: layer4، VGG16: features[24:]،
  EffNetB0: features[6:]) با lr=1e-5 (head: 1e-4)، حداکثر ۱۰ epoch، patience=4.
- **threshold تصمیم** روی val برای حداکثر balanced accuracy انتخاب و در چک‌پوینت ذخیره می‌شود.
- دلیل مرحله ۲: head-only روی pdraw_spiral دقت تست ~0.63–0.67 داشت (تقریباً شیر-یا-خط)؛
  با FT سبک AUC تست به 0.85–0.92 رسید. (تست شده و در پاسخ به کاربر گزارش شد.)

### نتایج آموزش (val AUC، از training_summary.csv و لاگ‌ها)

| dtype | model | val_auc_head | val_auc_ft | threshold |
|-------|-------|--------------|------------|-----------|
| pdraw_spiral | resnet50 | 1.000 | 1.000 | 0.50 |
| pdraw_wave | resnet50 | 1.000 | 1.000 | 0.55 |
| pdraw_wave | vgg16 | 0.918 | 0.939 | 0.45 |
| pdraw_wave | efficientnet_b0 | 1.000 | 1.000 | 0.35 |
| handpd_spiral | resnet50 | 0.986 | 0.967 | 0.95 |
| handpd_spiral | vgg16 | 0.938 | 0.960 | 0.45 |
| handpd_spiral | efficientnet_b0 | 0.967 | 0.981 | 0.50 |
| (بقیه در `checkpoints/training_summary.csv`) | | | | |

## ۵) تست‌های انجام‌شده و نتیجه

1. **corruptions smoke test** ✅ — هر ۶ خانواده × ۵ سطح روی یک تصویر؛ dtype/shape درست؛
   قطعیت RNG تأیید شد (`apply_corruption` دو بار → خروجی یکسان).
   گالری: `results/figures/corrupted_samples/` (۳۰ فایل).
2. **forward-pass هر ۳ مدل** ✅ — ابعاد ویژگی: ResNet50→2048، VGG16→512 (GAP روی نقشه کانولوشن)، EffNetB0→1280.
3. **Grad-CAM دستی** ✅ روی هر ۳ مدل (پس از دو اصلاح: نیاز به `x.requires_grad_()` چون backbone فروز است؛
   لایه هدف VGG16 = features[30] به‌خاطر خطای in-place ReLU).
   Spearman CAM تمیز vs بلور L5: ResNet50=0.868، VGG16=0.359، EffNet=0.386 (یافته‌ی جالب برای مقاله).
4. **inference تمیز** ✅ — نمونه: pdraw_spiral/resnet50 → acc=0.733, sens=1.000, spec=0.467, AUC=0.924.
   (توجه: sens/spec نامتوازن روی تست PDRAW به‌خاطر تفاوت توزیع split رسمی کگل است؛ AUC ملاک است.)
5. **robustness loop** ✅ (در جریان) — pdraw کامل (هر dtype: 93 ردیف = 3 مدل×31 شرط).

## ۶) نحوه‌ی اجرا (برای ادامه)

```bash
cd "C:\Users\ramti\Desktop\ramtin legion pc\article_project\src"

# ادغام نتایج بخش‌ها (پس از پایان ۴ پروسه robustness):
python merge_results.py            # -> results/robustness_results.csv (+predictions.csv)

# XAI (۴ پروسه موازی، ~۱۵ دقیقه):
for dt in pdraw_spiral pdraw_wave handpd_spiral handpd_meander; do
  OMP_NUM_THREADS=2 python run_xai.py --dtype $dt --out ../results/parts/xai_$dt.txt.csv > ../results/parts/xai_$dt.log 2>&1 &
done
# سپس دوباره merge (فایل xai_similarity.csv را هم می‌سازد)

# امتیاز و نمودارها:
python robustness_score.py         # -> results/robustness_scores.csv + breakdown
python make_figures.py             # -> results/figures/* + results/tables/* (CSV+LaTeX)
```

## ۷) کارهای باقی‌مانده (برای کاربر)

1. در `docs/paper/main.tex` جای `[Author 1 Name]` / `[Author 2 Name]` / `[email]`
   و `[repository URL]` نام‌های واقعی را بگذارید (تنها placeholderهای باقی‌مانده).
2. کامپایل مقاله: LaTeX روی این سیستم نصب نیست؛ فایل آماده است —
   `pdflatex main.tex` (دو بار برای cross-references) در Overleaf یا هر توزیع TeX.
   جداول از `results/tables/*_rows.tex` با `\input` خوانده می‌شوند، پس
   `results/` باید کنار `docs/` باشد (ساختار فعلی پروژه همین است).
3. (اختیاری) آپلود در GitHub طبق README.

## ۸) نکات مهم

- همه‌ی اسکریپت‌ها از داخل `src/` اجرا شوند (import های هم‌سطح).
- بازتولید کامل از صفر: `train.py` → `run_robustness.py` → `run_xai.py` (۴ dtype)
  → `merge_results.py` → `robustness_score.py` → `make_figures.py`
  (یا نسخه‌ی موازی با `--dtype`/`--out` طبق بخش ۶).
- اعداد مقاله فقط از `results/` می‌آیند؛ هیچ عددی دستی وارد نشده است.
- ناسازگاری sens/spec روی تست PDROW به تفاوت توزیع split رسمی کگل برمی‌گردد
  (در Discussion مقاله آمده)؛ AUC ملاک اصلی است.
