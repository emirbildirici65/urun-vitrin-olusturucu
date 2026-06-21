import gradio as gr
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from transformers import pipeline

# Ücretsiz en iyi arka plan temizleme modeli
print("Arka plan temizleme ve gölgelendirme motoru yükleniyor...")
rmbg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

def profesyonel_vitrin_uret(referans_resim, poz_secimi, golge_yogunlugu, isik_yonu):
    if referans_resim is None:
        return None
    
    # 1. ADIM: Ürünü Arka Plandan Ayır
    seffaf_urun = rmbg_pipe(referans_resim) # RGBA formatında ürün
    maske = rmbg_pipe(referans_resim, return_mask=True)
    
    # Ürünün sınırlarını bul ve kırp
    bbox = maske.getbbox()
    if bbox:
        seffaf_urun = seffaf_urun.crop(bbox)
        maske = maske.crop(bbox)
        
    # Tuval boyutunu e-ticaret standardı 1000x1000 yapıyoruz
    tuval_boyutu = (1000, 1000)
    
    # 2. ADIM: Poz ve Açı Ayarları (Gönderdiğin Referanslara Göre)
    if poz_secimi == "45° Yandan Poz (Kutulu/Kutusuz)":
        # Gönderdiğin 1. ve 3. görsellerdeki gibi hafif sağa eğimli derinlik hissi
        oran = 0.55
        seffaf_urun.thumbnail((int(tuval_boyutu[0]*oran), int(tuval_boyutu[1]*oran)), Image.Resampling.LANCZOS)
        # Kutulu veya yan poz için ürünü sol alt/orta odağa yerleştiriyoruz
        konum = (150, 450) 
        
    elif poz_secimi == "Düz Karşı Açı (Simetrik)":
        # Gönderdiğin 2. görseldeki gibi tam ortalanmış düz poz
        oran = 0.60
        seffaf_urun.thumbnail((int(tuval_boyutu[0]*oran), int(tuval_boyutu[1]*oran)), Image.Resampling.LANCZOS)
        konum = ((1000 - seffaf_urun.size[0]) // 2, 350)
        
    else: # Standart Katalog Pozu
        oran = 0.65
        seffaf_urun.thumbnail((int(tuval_boyutu[0]*oran), int(tuval_boyutu[1]*oran)), Image.Resampling.LANCZOS)
        konum = ((1000 - seffaf_urun.size[0]) // 2, (1000 - seffaf_urun.size[1]) // 2)

    # 3. ADIM: PROFESYONEL DOĞAL GÖLGE OLUŞTURMA (Soft Shadow)
    # Ürünün alt tabanına stüdyo gölgesi eklemek için maskeyi kullanıp siyah bir gölge katmanı üretiyoruz
    urun_maskesi = seffaf_urun.split()[-1]
    golge = Image.new("L", seffaf_urun.size, 0)
    
    # Gölge yoğunluğunu kullanıcı ayarına göre ayarlıyoruz
    golge_katmani = Image.new("RGBA", tuval_boyutu, (255, 255, 255, 255))
    saf_golge = Image.new("RGBA", seffaf_urun.size, (0, 0, 0, int(golge_yogunlugu)))
    
    # Gölgeyi yumuşatmak için Gaussian Blur uyguluyoruz (Doğal stüdyo ışığı hissi)
    bulanik_golge = Image.composite(saf_golge, Image.new("RGBA", seffaf_urun.size, (0,0,0,0)), urun_maskesi)
    bulanik_golge = bulanik_golge.filter(ImageFilter.GaussianBlur(radius=25))
    
    # Işık yönüne göre gölgeyi hafifçe kaydırıyoruz (Perspektif derinliği)
    if isik_yonu == "Sol Üstten (Sağa Gölge)":
        golge_konum = (konum[0] + 20, konum[1] + 35)
    elif isik_yonu == "Sağ Üstten (Sola Gölge)":
        golge_konum = (konum[0] - 20, konum[1] + 35)
    else: # Tam Üstten
        golge_konum = (konum[0], konum[1] + 40)

    # 4. ADIM: BİRLEŞTİRME (Arka Plan + Gölge + Ürün)
    beyaz_vitrin = Image.new("RGBA", tuval_boyutu, (255, 255, 255, 255))
    
    # Önce gölgeyi basıyoruz
    beyaz_vitrin.paste(bulanik_golge, golge_konum, bulanik_golge)
    # Üzerine net ürünü yerleştiriyoruz
    beyaz_vitrin.paste(seffaf_urun, konum, seffaf_urun)
    
    return beyaz_vitrin.convert("RGB")

# --- GRADIO KULLANICI ARAYÜZÜ ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray")) as demo:
    gr.Markdown("# ⚡ Profesyonel AI Vitrin Fotoğrafı Stüdyosu")
    gr.Markdown("Yüklediğiniz ham ürün görsellerini, stüdyo ışığı, yumuşak gölgelendirme ve tam istediğiniz açılarla profesyonel e-ticaret görseline dönüştürün.")
    
    with gr.Row():
        with gr.Column(scale=1):
            girdi_resim = gr.Image(label="Ham Ürün Fotoğrafı Yükle", type="pil")
            
            secim_poz = gr.Dropdown(
                choices=["Düz Karşı Açı (Simetrik)", "45° Yandan Poz (Kutulu/Kutusuz)", "Standart Katalog Pozu"],
                value="Düz Karşı Açı (Simetrik)",
                label="Hedef Vitrin Açısı / Poz Standardı"
            )
            
            secim_isik = gr.Radio(
                choices=["Tam Üstten (Dengeli)", "Sol Üstten (Sağa Gölge)", "Sağ Üstten (Sola Gölge)"],
                value="Tam Üstten (Dengeli)",
                label="Stüdyo Işık Yönü"
            )
            
            kaydirici_golge = gr.Slider(
                minimum=50, maximum=180, value=110, step=5,
                label="Zemin Gölge Belirginliği (Soft Shadow)"
            )
            
            buton_ucret = gr.Button("Vitrin Görseli Oluştur 🚀", variant="primary")
            
        with gr.Column(scale=1):
            cikti_resim = gr.Image(label="Oluşturulan Profesyonel Vitrin Fotoğrafı (1000x1000 HD)")
            
    buton_ucret.click(
        fn=profesyonel_vitrin_uret,
        inputs=[girdi_resim, secim_poz, kaydirici_golge, secim_isik],
        outputs=cikti_resim
    )

if __name__ == "__main__":
    demo.launch()
