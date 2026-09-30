# Folder Data

Berkas CSV harga tidak disertakan di repositori ini, dan diabaikan oleh
`.gitignore`.

## Cara mendapatkan data

```bash
python fetch_data.py --days 730
```

Berkas hasil unduhan akan tersimpan di folder ini sebagai
`prices_730.csv`.

## Format yang diharapkan

Bila ingin memakai data sendiri, susun CSV dengan format berikut lalu
letakkan di folder ini:

```
Tanggal,BTC,ETH
2024-09-29,65602.01,2657.62
2024-09-30,63327.59,2602.23
```

Aturannya:

- Kolom pertama bernama `Tanggal` atau `Date`, format `YYYY-MM-DD`
- Kolom `BTC` dan `ETH` berisi harga penutupan, desimal memakai titik
- Urutan tanggal dari lama ke baru, tanpa baris kosong
- Hapus pemisah ribuan pada angka

Aplikasi membaca seluruh berkas `.csv` di folder ini dan menampilkannya
sebagai pilihan pada sidebar.
