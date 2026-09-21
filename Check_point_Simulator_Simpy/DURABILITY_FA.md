# سن نسخهٔ کامل و ارزیابی بودجهٔ انتقال

دو فایل `evaluate_durability.py` و `test_durability.py` را در پوشهٔ simulator قرار دهید. کتابخانهٔ جدید لازم نیست و اجرای دوبارهٔ simulation هم لازم نیست؛ JSON و traceهای boundary_before/after استفاده می‌شوند.

```bash
python -m unittest test_durability -v
python evaluate_durability.py --run results/grouped_baseline/boundary_before.json --max-age-s 120 --startup-grace-s 45 --network-gbps 22.2 --store-gbps 500 --out results/grouped_baseline/durability_before.json
python evaluate_durability.py --run results/grouped_baseline/boundary_after.json --max-age-s 120 --startup-grace-s 45 --network-gbps 22.2 --store-gbps 500 --out results/grouped_baseline/durability_after.json
```

۱۲۰ ثانیه از configured_rpo این سناریو گرفته شده و اینجا به معنی حداکثر سن زمانی snapshot قابل بازیابی تفسیر می‌شود. ۴۵ ثانیه مهلت آغازین فقط یک فرض آزمایش است، نه SLA پروژه. برای ارزیابی سخت‌گیرانه از آغاز، `--startup-grace-s 0` بزنید؛ آن زمان بدون نسخهٔ کامل پنهان نمی‌شود. بودجهٔ شبکهٔ ۲۲٫۲ نیز همان فرض قبلی است، نه اندازه‌گیری fabric. ظرفیت ingest/store از حداقل NIC ورودی و دیسک store این سناریو (۵۰۰) آمده است.

نسخه فقط پس از رسیدن کل bytes همهٔ rankهای یک iteration کامل محسوب می‌شود. snapshot_s زمان منطقی مرز iteration است؛ ready_s آخرین پایان انتقال rankهاست. اگر نسخهٔ قدیمی دیرتر برسد نسخهٔ جدیدتر را جایگزین نمی‌کند. سن t-snapshot_s برای جدیدترین نسخهٔ کامل محاسبه می‌شود و تا پایان آموزش هر job ارزیابی می‌شود. زمان قبل از نخستین نسخهٔ کامل جدا گزارش می‌شود. هیچ checkpoint اولیه فرض نشده است. این معیار age با تعداد iterationهای rollback و زمان recovery برابر نیست.

در اجرای محلی، قبل از تغییر فاز نخستین نسخهٔ کامل برای همهٔ jobها در ۴۰٫۸۱ ثانیه ساخته شد. پس از تغییر، برای alpha0 و be0 به ۲۷٫۵۷ رسید و برای bravo0 همان ۴۰٫۸۱ ماند. سن بیشینه بعد از اولین نسخه حدود ۴۰٫۸۹ ثانیه بود. میانگین‌های age دورهٔ مشاهدهٔ متفاوت دارند، زیرا آغاز protected interval تغییر کرده است؛ آن‌ها را بدون یکسان‌سازی بازهٔ مشاهده، به‌عنوان بهبود/افت قطعی مقایسه نکنید.

پروفایل ترافیک با تقسیم حجم هر انتقال بر مدت همان انتقال ساخته می‌شود؛ این تخمین نرخ یکنواخت است، نه trace نرخ لحظه‌ای. `necessary_peak_lower_bound_gbps` از جمع bytes انتقال‌هایی به دست می‌آید که دقیقاً بازهٔ زمانی مشترک دارند. اگر این کران از بودجه بیشتر باشد، انجام همین bytes در همین بازه‌ها با آن بودجه غیرممکن است. پایین‌تر بودن کران از بودجه تضمین کافی نیست؛ سایر پنجره‌ها و تغییر نرخ‌ها باید بررسی شوند. خروجی `not_disproved_not_certified` یعنی بودجه با این آزمون رد نشده ولی تأیید نشده است.

در هر دو اجرای فعلی peak proxy و کران پایین برابر ۱۲ GB/s شدند. در optimizer اولیه peak برابر ۳ بود: سه job با سقف یک GB/s برای هر job. simulator دوازده rank با سقف یک GB/s برای هر stream دارد. این تفاوت به معنی خطای جمع ساده نیست؛ مدل باید concurrency/stream و سقف‌های pool را با runtime یکسان کند. کاهش overlap از ۳۰ به ۲۲ اوج هم‌زمانی را حذف نکرده است.

می‌توانید همان trace را با `--network-gbps 3` و یک فایل خروجی جدا بررسی کنید. این فقط بررسی خلاف‌واقع روی بازه‌های ثبت‌شده است، نه شبیه‌سازی مجدد با شبکهٔ محدودتر؛ verdict باید ناممکن بودن این بازه‌ها زیر سقف ۳ را گزارش کند. simulator با بودجهٔ کمتر ممکن است زمان انتقال‌ها را بلندتر کند.

این ابزار برای direct-store بدون خرابی است. بازسازی peer، parity، failure recovery، اثبات ظرفیت لحظه‌ای و بهینه‌سازی مشترک فرکانس/فاز در آن پیاده نشده‌اند. بعد از تطبیق مدل، معیار age باید در انتخاب گزینه‌ها و بازاجرای simulator اعمال شود؛ فیلتر پس از اجرا به‌تنهایی optimizer را constrained نمی‌کند.
