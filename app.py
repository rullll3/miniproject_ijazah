
import re
import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

# Jika Tesseract tidak ditemukan, hapus tanda komentar
# pada dua baris berikut dan sesuaikan lokasi instalasinya.
pytesseract.pytesseract.tesseract_cmd = (
r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

st.set_page_config(
    page_title="Mini Project Ijazah",
    page_icon="📄",
    layout="wide"
)

st.title("Deteksi Nomor Ijazah dan Tanda Tangan")
st.write(
    "Pengolahan citra ijazah menggunakan OpenCV "
    "dan pembacaan teks menggunakan Tesseract OCR."
)

uploaded_file = st.file_uploader(
    "Unggah gambar ijazah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ==========================================
    # TAHAP 1: INPUT CITRA IJAZAH
    # ==========================================
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)
    height, width = original.shape[:2]

    st.subheader("1. Citra Ijazah")
    st.image(original, caption="Citra asli", width="stretch")

    # ==========================================
    # TAHAP 2: GRAYSCALE
    # ==========================================
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    st.subheader("2. Grayscale")
    st.image(gray, caption="Citra grayscale",
             clamp=True, width="stretch")

    # ==========================================
    # TAHAP 3: IMAGE ENHANCEMENT
    # CLAHE
    # ==========================================
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    st.subheader("3. Image Enhancement")
    st.image(enhanced, caption="Hasil CLAHE",
             clamp=True, width="stretch")

    st.divider()
    st.subheader("4. Tentukan Area Pemrosesan")

    st.write(
        "Atur koordinat area nomor dan tanda tangan "
        "berdasarkan posisi pada gambar asli. "
        "Nilai koordinat menggunakan persentase."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Area Nomor Ijazah**")
        nx = st.slider(
            "Rentang X nomor (%)", 0, 100, (40, 95)
        )
        ny = st.slider(
            "Rentang Y nomor (%)", 0, 100, (50, 90)
        )

    with col2:
        st.markdown("**Area Tanda Tangan**")
        sx = st.slider(
            "Rentang X tanda tangan (%)", 0, 100, (40, 95)
        )
        sy = st.slider(
            "Rentang Y tanda tangan (%)", 0, 100, (50, 95)
        )

    def crop_roi(image, x_range, y_range):
        h, w = image.shape[:2]

        x1 = int(w * x_range[0] / 100)
        x2 = int(w * x_range[1] / 100)
        y1 = int(h * y_range[0] / 100)
        y2 = int(h * y_range[1] / 100)

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2]

    nomor_roi = crop_roi(enhanced, nx, ny)
    tanda_roi = crop_roi(enhanced, sx, sy)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            nomor_roi,
            caption="ROI area nomor",
            clamp=True,
            width="stretch"
        )

    with col2:
        st.image(
            tanda_roi,
            caption="ROI area tanda tangan",
            clamp=True,
            width="stretch"
        )

    st.info(
        "Pastikan kedua potongan benar-benar mencakup "
        "area nomor dan tanda tangan yang ingin diperiksa."
    )

    # ==========================================
    # PEMROSESAN PIPELINE
    # ==========================================
    if st.button("Jalankan Pipeline", type="primary"):

        # ======================================
        # CABANG A: AREA NOMOR
        # Enhancement -> OCR
        # ======================================
        st.divider()
        st.header("Cabang A: Pembacaan Nomor Ijazah")

        # Enhancement tambahan dengan memperbesar area
        nomor_besar = cv2.resize(
            nomor_roi,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        # Thresholding tambahan untuk membantu OCR
        _, nomor_binary = cv2.threshold(
            nomor_besar,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        st.image(
            nomor_binary,
            caption="Area nomor setelah enhancement "
                    "dan thresholding",
            clamp=True,
            width="stretch"
        )

        try:
            hasil_ocr = pytesseract.image_to_string(
                nomor_binary,
                lang="eng",
                config="--psm 6"
            )

            hasil_ocr = re.sub(
                r"[^A-Za-z0-9./\-\s]",
                "",
                hasil_ocr
            ).strip()

            if hasil_ocr:
                st.success("Hasil OCR: " + hasil_ocr)
            else:
                st.warning(
                    "Nomor belum terbaca. Periksa ROI "
                    "atau kualitas citra."
                )

        except Exception as error:
            hasil_ocr = ""
            st.error("OCR gagal dijalankan.")
            st.code(str(error))

        # ======================================
        # CABANG B: AREA TANDA TANGAN
        # Thresholding -> Morphology -> Detection
        # ======================================
        st.divider()
        st.header("Cabang B: Deteksi Tanda Tangan")

        # Thresholding: objek gelap menjadi putih
        _, tanda_binary = cv2.threshold(
            tanda_roi,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        st.image(
            tanda_binary,
            caption="Hasil thresholding tanda tangan",
            clamp=True,
            width="stretch"
        )

        # Morphology untuk mengurangi bintik kecil
        kernel = np.ones((2, 2), np.uint8)

        tanda_bersih = cv2.morphologyEx(
            tanda_binary,
            cv2.MORPH_OPEN,
            kernel
        )

        st.image(
            tanda_bersih,
            caption="Hasil morphology opening",
            clamp=True,
            width="stretch"
        )

        # Menghitung piksel objek setelah morphology
        jumlah_tinta = np.count_nonzero(tanda_bersih)
        total_piksel = tanda_bersih.size
        rasio_tinta = jumlah_tinta / total_piksel

        batas = st.slider(
            "Ambang minimum piksel objek (%)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1
        )

        persen_tinta = rasio_tinta * 100

        st.metric(
            "Persentase piksel objek",
            f"{persen_tinta:.2f}%"
        )

        tanda_terdeteksi = persen_tinta >= batas

        if tanda_terdeteksi:
            st.success(
                "Coretan terdeteksi pada area tanda tangan."
            )
        else:
            st.warning(
                "Coretan tidak terdeteksi berdasarkan ambang."
            )

        st.caption(
            "Deteksi berdasarkan piksel belum membuktikan "
            "keaslian tanda tangan. Stempel, teks, dan noda "
            "juga dapat dianggap sebagai coretan."
        )

        # ======================================
        # HASIL VERIFIKASI
        # ======================================
        st.divider()
        st.header("Hasil Verifikasi")

        st.write("**Nomor ijazah:**")
        st.write(hasil_ocr if hasil_ocr else "Belum terbaca")

        st.write("**Keberadaan tanda tangan:**")
        st.write(
            "Terdeteksi"
            if tanda_terdeteksi
            else "Tidak terdeteksi"
        )

else:
    st.info("Unggah gambar ijazah untuk memulai.")