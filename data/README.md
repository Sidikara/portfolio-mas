## Penggunaan

Data harga sudah disertakan di folder `data/`, jadi langsung bisa dijalankan:

    streamlit run app.py
    
Aturannya:

- Kolom pertama bernama `Tanggal` atau `Date`, format `YYYY-MM-DD`
- Kolom `BTC` dan `ETH` berisi harga penutupan, desimal memakai titik
- Urutan tanggal dari lama ke baru, tanpa baris kosong
- Hapus pemisah ribuan pada angka

Aplikasi membaca seluruh berkas `.csv` di folder ini dan menampilkannya
sebagai pilihan pada sidebar.
