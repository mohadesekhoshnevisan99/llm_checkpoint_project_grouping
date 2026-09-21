# ارزیابی فاز با مرز واقعی iteration و بازاجرای simulator

بسته شامل نسخهٔ جدید `run_direct_contract.py`، ابزار جدید `optimize_boundary_phases.py` و دو فایل تست است. فایل‌ها را در پوشهٔ `Check_point_Simulator_Simpy` قرار دهید؛ runner قبلی را با نسخهٔ جدید جایگزین کنید. فایل‌های اصلی simulator تغییر نمی‌کنند. ابزار جست‌وجو فقط کتابخانهٔ استاندارد Python می‌خواهد؛ runner از محیط simulator استفاده می‌کند.

## اجرا در WSL

```bash
cd /home/mohadeseh/sampaper/Check_point_Simulator_Simpy
source .venv/bin/activate
python -m unittest test_direct_contract test_boundary_phases -v
```

شش آزمون باید بگذرد. آزمون integration در صورت پیدا نکردن مخزن simulator ممکن است skip شود؛ اجرای CLI زیر تست روی پروژهٔ شماست. تست integration فایل‌های خودش را در `outputs/boundary_validation` می‌نویسد، نه فایل‌های baseline شما.

۱. اجرای مبنا با ثبت تمام مرزهای iteration؛ فایل قدیمی اطلاعات کافی برای جست‌وجو ندارد:

```bash
python run_direct_contract.py --scenario scenarios/grouped_handcheck_no_failures.yaml --contract results/grouped_baseline/optimizer_policy_contract.json --out results/grouped_baseline/boundary_before.json
```

۲. جست‌وجوی مشترک فازها روی مرزهای ثبت‌شده:

```bash
python optimize_boundary_phases.py --run results/grouped_baseline/boundary_before.json --phase-step 1 --out-contract results/grouped_baseline/boundary_candidate_contract.json --report results/grouped_baseline/boundary_search.json
```

۳. بازاجرای واقعی با قرارداد پیشنهادی:

```bash
python run_direct_contract.py --scenario scenarios/grouped_handcheck_no_failures.yaml --contract results/grouped_baseline/boundary_candidate_contract.json --out results/grouped_baseline/boundary_after.json
```

۴. مقایسهٔ دو اجرای واقعی:

```bash
python optimize_boundary_phases.py --run results/grouped_baseline/boundary_before.json --compare-run results/grouped_baseline/boundary_after.json --report results/grouped_baseline/boundary_comparison.json
```

## چه چیزی تغییر کرده است؟

runner اکنون همهٔ مرزها را ثبت می‌کند، حتی زمانی که checkpoint نمی‌گیرد. شروع و پایان ارسال rankها به store، برای هر موج job در یک بازه از نخستین شروع تا آخرین پایان جمع‌بندی می‌شود.

جست‌وجو فازهای یک‌ثانیه‌ای و فاز مبنا را بررسی می‌کند. هر موعد به اولین مرز مجاز بعد از آن نگاشت می‌شود؛ با زمان capture و انتقال مشاهده‌شده، busy gating و حذف موعدهای سپری‌شده هم بازسازی می‌شوند. بیشترین مدت capture-to-store و انتقال مشاهده‌شدهٔ هر job به‌عنوان تخمین محافظه‌کارانه برای همین trace استفاده می‌شود؛ کران معتبر برای همهٔ اجراهای آینده نیست.

هدف به ترتیب lexicographic: کمترین مجموع زمان هم‌پوشانی جفت انتقال‌های jobها؛ سپس کمترین بیشینهٔ تعداد jobهای هم‌زمان در حال انتقال؛ سپس کمترین مجموع تأخیر موعد تا مرز. برای سه job هم‌زمان به مدت یک ثانیه، overlap برابر سه pair-second است، نه سه ثانیهٔ wall-clock جداگانه. این شاخص میزان مصرف دقیق لینک نیست و صفر کردن آن لزوماً هدف بهینهٔ throughput نیست.

برای جلوگیری از کاهش ظاهری برخورد از طریق حذف checkpoint، گزینه باید تعداد موج یکسان، بیشترین فاصلهٔ capture بدون افزایش، نخستین capture بدون تأخیر بیشتر، و آخرین capture بدون جلو افتادن داشته باشد. این قیدها روی پیش‌بینی با مرزهای ثابت هستند؛ نتیجه باید دوباره در simulator اجرا شود.

دوره‌ها در این نسخه ثابت می‌مانند. تغییر هم‌زمان فرکانس نیازمند هزینهٔ معتبر خرابی/rollback و معیار freshness است؛ افزودن آن به آزمایش بدون خرابی با هدف صرفاً کاهش overlap می‌تواند به حذف checkpoint منجر شود. گروه‌ها، route و donorها تغییر نمی‌کنند؛ فقط قراردادهای direct فعلی پشتیبانی می‌شوند.

تا ۲۰۰ هزار ترکیب، جست‌وجوی کامل در فهرست فازهای فیلترشده انجام می‌شود؛ بالاتر از آن coordinate descent با سقف پنج دور است و تضمین سراسری ندارد. تعداد فازها و دامنهٔ trace نیز جست‌وجو را محدود می‌کنند.

## نتیجهٔ محلی

روی همان handcheck بدون خرابی، ۲۱۶ ترکیب بررسی شد. فازهای alpha0=23، be0=25 و bravo0=28 پیشنهاد شدند. overlap پیش‌بینی‌شده و مشاهده‌شده در بازاجرا از ۳۰ به ۲۲ pair-second رسید. هر job همچنان ۱۰ موج داشت؛ کل نوشتن store برابر ۱۲۰ GB و پایان آموزش ۳۱۸٫۶ ثانیه ماند. بیشترین تعداد jobهای هم‌زمان همچنان ۳ بود. بیشترین lag اندکی از ۱۲٫۷۲۹۷ به ۱۲٫۸۰۵۴ ثانیه افزایش یافت؛ آن معیار هدف اصلی جست‌وجو نیست. کاهش ۲۶٫۷ درصدی overlap به معنی افزایش ۲۶٫۷ درصدی goodput نیست.

بخش `replay_checks` تعداد موج، فاصلهٔ بیشینه، حجم store، overlap و پایان آموزش را مقایسه می‌کند. `passes_this_limited_replay_check` فقط مجموعهٔ همین آزمون‌هاست؛ reliability، حداکثر age نسخهٔ کامل، بودجهٔ fabric، بارهای دیگر و startup را اثبات نمی‌کند. `optimizer_calendar_verified` و `capacity_certified` همچنان false هستند.

مرزهای iteration خودشان به policy وابسته‌اند. جست‌وجوی روی trace ثابت صرفاً پیشنهاد می‌دهد؛ بازاجرای simulator تصمیم را راستی‌آزمایی می‌کند. برای workloadهای دیگر ممکن است پیشنهاد بهتر نشود یا قیدهای پیش‌بینی‌شده در بازاجرا نقض شوند. در آن حالت قرارداد مبنا را نگه دارید و با trace تازه بازمدل‌سازی کنید؛ نتیجهٔ پیش‌بینی را به‌عنوان نتیجهٔ واقعی گزارش نکنید.
