import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────
# 1. SETUP DASH APP & CUSTOM STYLING (Tema Premium)
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.LUMEN, 
        "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

app.title = "Dashboard IPM Jawa Timur"

server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { font-family: 'Poppins', sans-serif; background-color: #f4f7fe; color: #1e293b; margin: 0; }
            .sidebar { position: fixed; top: 0; left: 0; bottom: 0; width: 260px; background-color: #ffffff; border-right: 1px solid #e2e8f0; padding: 25px 20px; z-index: 1000; }
            .main-content { margin-left: 260px; padding: 30px; min-height: 100vh; }
            .topbar { background: white; padding: 15px 30px; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.02); display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; border: 1px solid rgba(226, 232, 240, 0.6); }
            .card-custom { background: white; border-radius: 20px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.03); border: 1px solid rgba(226, 232, 240, 0.6); margin-bottom: 24px; transition: all 0.3s ease; }
            .card-custom:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(2, 132, 199, 0.08); border-color: rgba(2, 132, 199, 0.2); }
            .kpi-title { font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;}
            .kpi-val { font-size: 34px; font-weight: 700; color: #1e293b; line-height: 1.2; margin-bottom: 4px; }
            .kpi-sub { font-size: 12px; color: #94a3b8; font-weight: 500; }
            .sec-title { font-size: 18px; font-weight: 600; color: #1e293b; margin-bottom: 15px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; display: inline-block; }
            .insight-box { background: rgba(224, 242, 254, 0.4); border-left: 4px solid #0284c7; padding: 15px 20px; border-radius: 0 12px 12px 0; font-size: 13.5px; color: #0369a1; margin-top: 15px; line-height: 1.6; }
            .warn-box { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 15px 20px; border-radius: 0 12px 12px 0; font-size: 13.5px; color: #78350f; margin-top: 15px; line-height: 1.6; }
            .success-box { background: #f0fdf4; border-left: 4px solid #10b981; padding: 15px 20px; border-radius: 0 12px 12px 0; font-size: 13.5px; color: #14532d; margin-top: 15px; line-height: 1.6; }
            .danger-box { background: #fef2f2; border-left: 4px solid #ef4444; padding: 15px 20px; border-radius: 0 12px 12px 0; font-size: 13.5px; color: #7f1d1d; margin-top: 15px; line-height: 1.6; }
            /* --- DARK MODE MAGIC --- */
            .dark-mode { background-color: #0f172a !important; color: #f8fafc !important; }
            .dark-mode .sidebar { background-color: #1e293b !important; border-color: #334155 !important; }
            .dark-mode .main-content { background-color: #0f172a !important; }
            .dark-mode .topbar { background-color: #1e293b !important; border-color: #334155 !important; color: #f8fafc !important; }
            .dark-mode .topbar h4, .dark-mode .topbar p { color: #f8fafc !important; }
            .dark-mode .card-custom { background-color: #1e293b !important; border-color: #334155 !important; box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important; color: #f8fafc !important;}
            .dark-mode .sec-title { color: #f8fafc !important; border-color: #334155 !important; }
            .dark-mode .kpi-val, .dark-mode .kpi-title { color: #f8fafc !important; }
            .dark-mode .nav-pills .nav-link { color: #94a3b8 !important; }
            .dark-mode .nav-pills .nav-link.active { background-color: #3b82f6 !important; color: white !important; }
            .dark-mode .insight-box { background: rgba(56, 189, 248, 0.1) !important; border-color: #38bdf8 !important; color: #bae6fd !important; }
            .dark-mode .warn-box { background: rgba(245, 158, 11, 0.1) !important; color: #fcd34d !important; }
            .dark-mode .success-box { background: rgba(16, 185, 129, 0.1) !important; color: #6ee7b7 !important; }
            .dark-mode .danger-box { background: rgba(239, 68, 68, 0.1) !important; color: #fca5a5 !important; }
            
            /* Penyesuaian Tabel & Dropdown untuk Dark Mode */
            .dark-mode .Select-control, .dark-mode .Select-menu-outer { background-color: #334155 !important; border-color: #475569 !important; color: white !important; }
            .dark-mode .Select-value-label { color: white !important; }
            .dark-mode label {background-color: transparent !important;}
            .dark-mode .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th { background-color: #334155 !important; color: white !important; border-color: #475569 !important; }
            .dark-mode .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td { background-color: #1e293b !important; color: white !important; border-color: #475569 !important; }
            /* Perbaikan label slider di dark mode */
            .dark-mode .rc-slider-mark-text { color: #94a3b8 !important; }
            .dark-mode .rc-slider-mark-text-active { color: #f8fafc !important; }

            /* Perbaikan agar teks di dalam info-box tetap kontras */
            .dark-mode .info-box { color: #f8fafc !important; }
            .teks-stats { color: #1e293b; }
            .dark-mode .teks-stats { color: #f8fafc !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ─────────────────────────────────────────────
# 2. LOAD DATA & PREPROCESSING
# ─────────────────────────────────────────────
PROV_TREND = {
    "years": [2019, 2020, 2021, 2022, 2023, 2024],
    "ipm": [71.50, 71.71, 72.14, 72.75, 73.38, 74.05], 
}

try:
    df_master = pd.read_csv("data_jatim.csv", sep=",", encoding="utf-8-sig")
    df_master.columns = df_master.columns.str.strip()
    if len(df_master.columns) == 1:
        df_master = pd.read_csv("data_jatim.csv", sep=";", encoding="utf-8-sig")
        df_master.columns = df_master.columns.str.strip()
except Exception as e:
    df_master = pd.DataFrame() 

list_tahun = sorted(df_master['Tahun'].dropna().unique().tolist(), reverse=True) if not df_master.empty else [2024]
list_kawasan = ["Semua Kawasan"] + df_master['Kawasan'].dropna().unique().tolist() if not df_master.empty else ["Semua Kawasan"]
list_kab = sorted(df_master['Kabupaten_Kota'].dropna().unique().tolist()) if not df_master.empty else []

opsi_indikator = {
    "Umur Harapan Hidup (Kesehatan)": "UHH",
    "Harapan Lama Sekolah (Pendidikan)": "HLS",
    "Rata-rata Lama Sekolah (Pendidikan)": "RLS",
    "Angka Partisipasi Sekolah SMA": "APS_SMA"
}

import os
import json
import urllib.request

# ─────────────────────────────────────────────
# 3. LAYOUT UTAMA (Sidebar L-Shape & Top Navbar)
# ─────────────────────────────────────────────
sidebar = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # --- BAGIAN ATAS (Logo & Menu) ---
    html.Div([
        html.Div([
            html.H3([html.Span("JATIM", style={"color": "#0284c7"}), html.Span("STATS", className="teks-stats", style={"fontWeight": "400"})], style={"fontWeight": "800", "letterSpacing": "1px", "marginBottom": "40px"})        ]),
        html.Div([
            html.P("MENU UTAMA", style={"fontSize": "11px", "color": "#94a3b8", "fontWeight": "600", "letterSpacing": "1px", "marginBottom": "15px"}),
            dbc.Nav([
                dbc.NavLink([html.I(className="fa-solid fa-chart-pie me-3"), "Dashboard"], href="/", id="link-dashboard", active="exact", style={"fontWeight": "500", "color": "#64748b", "padding": "12px 15px", "marginBottom": "10px", "borderRadius": "12px"}),
                dbc.NavLink([html.I(className="fa-solid fa-table me-3"), "Dataset"], href="/manage", id="link-manage", active="exact", style={"fontWeight": "500", "color": "#64748b", "padding": "12px 15px", "marginBottom": "10px", "borderRadius": "12px"}),
                dbc.NavLink([html.I(className="fa-solid fa-book-open me-3"), "Kamus & Eksplorasi Data"], href="/metadata", id="link-metadata", active="exact", style={"fontWeight": "500", "color": "#64748b", "padding": "12px 15px", "marginBottom": "10px", "borderRadius": "12px"}),
            ], vertical=True, pills=True)
        ])
    ]),
    
    # --- BAGIAN BAWAH (Toggle Dark/Light Mode) ---
    html.Div([
        html.Hr(style={"borderColor": "#e2e8f0", "marginBottom": "15px"}),
        html.Div([
            # Memori untuk menyimpan status tema
            dcc.Store(id='theme-store', data='light'),
            html.Div(id='theme-icon', children=[html.I(className="fa-solid fa-sun fa-lg", style={"color": "#f59e0b"})], style={"width": "30px", "textAlign": "center"}),
            html.Span("Light Mode", id='theme-text', style={"fontWeight": "600", "fontSize": "13px", "marginLeft": "10px", "color": "#64748b"})
        ], id='theme-toggle-btn', n_clicks=0, style={"display": "flex", "alignItems": "center", "cursor": "pointer", "padding": "10px 15px", "borderRadius": "12px", "backgroundColor": "#f8fafc", "border": "1px solid #e2e8f0", "transition": "all 0.3s ease"})
    ], style={"marginTop": "auto"}) # Ini yang bikin dia didorong mentok ke bawah!

], className="sidebar", style={"display": "flex", "flexDirection": "column", "height": "100vh"}) # Tambahkan flex agar atas-bawah bekerja

topbar = html.Div([
    html.Div([
        html.H4("Profil Pembangunan Jawa Timur", style={"margin": "0", "fontWeight": "700", "color": "#1e293b", "fontSize": "22px"}),
        html.P("Statistik Kesejahteraan Rakyat dan Ekonomi Regional (2019-2024)", style={"margin": "0", "fontSize": "13px", "color": "#64748b"})
    ]),
], className="topbar")

# Panel Filter Global
filter_panel = html.Div([
    dbc.Row([
        dbc.Col([
            html.Label("📅 Pilih Tahun", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "8px"}),
            dcc.Dropdown(id='filter-tahun', options=[{'label': str(t), 'value': t} for t in list_tahun], value=list_tahun[0], clearable=False, style={"borderRadius": "10px"})
        ], width=3),
        dbc.Col([
            html.Label("🗺️ Pilih Kawasan", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "8px"}),
            dcc.Dropdown(id='filter-kawasan', options=[{'label': k, 'value': k} for k in list_kawasan], value="Semua Kawasan", clearable=False, style={"borderRadius": "10px"})
        ], width=3),
        dbc.Col([
            html.Label("📍 Filter Kabupaten/Kota", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "8px"}),
            dcc.Dropdown(id='filter-kab', options=[{'label': k, 'value': k} for k in list_kab], value=list_kab, multi=True, placeholder="Pilih wilayah...", style={"borderRadius": "10px"})
        ], width=6),
    ])
], className="card-custom", style={"padding": "20px 24px", "borderTop": "4px solid #0284c7"})

# Tabs Navigation
tabs_menu = html.Div([
    dcc.RadioItems(
        id='tabs-selector',
        options=[
            {'label': ' 🏠 Executive Summary', 'value': 'tab-1'},
            {'label': ' 🔍 Dekonstruksi IPM', 'value': 'tab-2'},
            {'label': ' ⚖️ Ekonomi & Ketimpangan', 'value': 'tab-3'},
            {'label': ' 🤖 Analisis Klaster', 'value': 'tab-4'},
        ],
        value='tab-1',
        inline=True,
        inputStyle={"display": "none"},
        labelStyle={
            "cursor": "pointer", "padding": "12px 20px", "marginRight": "12px", 
            "backgroundColor": "white", "borderRadius": "12px", "fontWeight": "600",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.02)", "border": "1px solid #e2e8f0",
            "fontSize": "14px", "color": "#64748b", "transition": "all 0.2s ease"
        }
    )
], style={"marginBottom": "25px"})

app.layout = html.Div(id='app-wrapper', className='', children=[
    sidebar,
    html.Div([
        topbar,
        html.Div(id='page-content') # Konten halaman masuk sini
    ], className="main-content")
])

# ---------------------------------------------------------
# CALLBACK NAVIGASI HALAMAN (Router)
# ---------------------------------------------------------
@app.callback(
    Output('page-content', 'children'),
    Output('link-dashboard', 'style'),
    Output('link-manage', 'style'),
    Output('link-metadata', 'style'), # <--- TAMBAHAN OUTPUT BARU
    Input('url', 'pathname')
)
def display_page(pathname):
    inactive_style = {"fontWeight": "500", "color": "#64748b", "padding": "12px 15px", "marginBottom": "10px", "borderRadius": "12px", "backgroundColor": "transparent"}
    active_style = {"fontWeight": "600", "color": "#0284c7", "backgroundColor": "#e0f2fe", "borderRadius": "12px", "padding": "12px 15px", "marginBottom": "10px"}

    # ---------------------------------------------------------
    # HALAMAN 1: KAMUS & PROFIL DATA (BARU)
    # ---------------------------------------------------------
    if pathname == '/metadata':
        # 1. Data Kamus Variabel (Statis)
        data_kamus = [
            {"Variabel": "Tahun", "Deskripsi": "Menunjukkan periode waktu pengamatan data pembangunan manusia dan ekonomi di Jawa Timur.", "Satuan / Bentuk Data": "Tahun"},
            {"Variabel": "Kabupaten/Kota", "Deskripsi": "Menunjukkan nama wilayah administratif yang menjadi unit analisis.", "Satuan / Bentuk Data": "Kategori"},
            {"Variabel": "Kawasan", "Deskripsi": "Menunjukkan kelompok kawasan fungsional wilayah (misalnya GKS, Madura, Tapal Kuda, dll).", "Satuan / Bentuk Data": "Kategori"},
            {"Variabel": "IPM", "Deskripsi": "Indeks Pembangunan Manusia yang menggambarkan kualitas hidup masyarakat berdasarkan pendidikan, kesehatan, dan standar hidup layak.", "Satuan / Bentuk Data": "Indeks"},
            {"Variabel": "UHH", "Deskripsi": "Umur Harapan Hidup, menggambarkan rata-rata perkiraan lama hidup penduduk.", "Satuan / Bentuk Data": "Tahun"},
            {"Variabel": "HLS", "Deskripsi": "Harapan Lama Sekolah, menunjukkan lamanya pendidikan formal yang diharapkan akan ditempuh penduduk usia sekolah.", "Satuan / Bentuk Data": "Tahun"},
            {"Variabel": "RLS", "Deskripsi": "Rata-rata Lama Sekolah, menunjukkan rata-rata jumlah tahun pendidikan yang telah ditempuh penduduk.", "Satuan / Bentuk Data": "Tahun"},
            {"Variabel": "PDRB Per Kapita", "Deskripsi": "Nilai rata-rata output ekonomi per penduduk di suatu wilayah. Digunakan untuk melihat tingkat kesejahteraan ekonomi daerah.", "Satuan / Bentuk Data": "Juta Rupiah"},
            {"Variabel": "Persentase Miskin", "Deskripsi": "Persentase penduduk miskin di suatu wilayah. Menunjukkan tingkat kerentanan ekonomi masyarakat.", "Satuan / Bentuk Data": "Persen (%)"},
            {"Variabel": "Gini Ratio", "Deskripsi": "Indikator ketimpangan distribusi pendapatan di suatu wilayah.", "Satuan / Bentuk Data": "Rasio"},
            {"Variabel": "TPT", "Deskripsi": "Tingkat Pengangguran Terbuka.", "Satuan / Bentuk Data": "Persen (%)"}
        ]

        # 2. Data Statistika Deskriptif (Dihitung Otomatis Dinamis)
        interpretasi_dict = {
            "IPM": "Sebagian besar daerah berada pada tingkat pembangunan menengah-tinggi.",
            "UHH": "Variasi kesehatan relatif kecil antarwilayah.",
            "RLS": "Kesenjangan pendidikan masih terlihat cukup jelas.",
            "HLS": "Harapan pendidikan relatif baik di mayoritas wilayah.",
            "PDRB_Per_Kapita": "Terdapat kesenjangan ekonomi yang sangat tinggi antarwilayah.",
            "Persentase_Miskin": "Tingkat kemiskinan masih cukup bervariasi.",
            "Gini_Ratio": "Ketimpangan pendapatan relatif moderat.",
            "TPT": "Pengangguran antarwilayah masih cukup beragam."
        }
        
        data_stats = []
        if not df_master.empty:
            kolom_numerik = ['IPM', 'UHH', 'RLS', 'HLS', 'PDRB_Per_Kapita', 'Persentase_Miskin', 'Gini_Ratio', 'TPT']
            for col in kolom_numerik:
                if col in df_master.columns:
                    data_stats.append({
                        "Variabel": col,
                        "Rata-rata": round(df_master[col].mean(), 2),
                        "Median": round(df_master[col].median(), 2),
                        "Minimum": round(df_master[col].min(), 2),
                        "Maksimum": round(df_master[col].max(), 2),
                        "Std. Deviasi": round(df_master[col].std(), 2),
                        "Interpretasi Singkat": interpretasi_dict.get(col, "-")
                    })

        page_meta = html.Div([
            html.Div([
                html.Div("📖 Kamus & Profil Data", className="sec-title", style={"borderBottom": "none", "marginBottom": "0"}),
                html.Div("Metadata & Ringkasan Statistik", style={"fontSize": "13px", "color": "#64748b", "fontWeight": "500"})
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}),
            
            # Tabel 1: Kamus Data
            html.Div([
                html.Div("Kamus Variabel (Metadata)", className="sec-title", style={"fontSize": "16px"}),
                dash_table.DataTable(
                    data=data_kamus,
                    columns=[{'name': i, 'id': i} for i in ["Variabel", "Deskripsi", "Satuan / Bentuk Data"]],
                    style_table={'overflowX': 'auto', 'borderRadius': '10px', 'border': '1px solid #e2e8f0', 'marginBottom': '30px'},
                    style_header={'backgroundColor': '#f8fafc', 'fontWeight': 'bold', 'color': '#1e293b', 'padding': '12px'},
                    style_cell={'textAlign': 'left', 'padding': '12px', 'fontFamily': 'Poppins', 'fontSize': '13px'},
                    style_data={'whiteSpace': 'normal', 'height': 'auto'}, # Agar teks deskripsi bisa wrap/turun ke bawah
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}]
                ),
                
                # Tabel 2: Statistika Deskriptif
                html.Div("Statistika Deskriptif (Keseluruhan Tahun)", className="sec-title", style={"fontSize": "16px"}),
                dash_table.DataTable(
                    data=data_stats,
                    columns=[{'name': i, 'id': i} for i in ["Variabel", "Rata-rata", "Median", "Minimum", "Maksimum", "Std. Deviasi", "Interpretasi Singkat"]],
                    style_table={'overflowX': 'auto', 'borderRadius': '10px', 'border': '1px solid #e2e8f0'},
                    style_header={'backgroundColor': '#f8fafc', 'fontWeight': 'bold', 'color': '#1e293b', 'padding': '12px'},
                    style_cell={'textAlign': 'left', 'padding': '12px', 'fontFamily': 'Poppins', 'fontSize': '13px'},
                    style_data={'whiteSpace': 'normal', 'height': 'auto'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}]
                )
            ], className="card-custom")
        ])
        return page_meta, inactive_style, inactive_style, active_style

    # ---------------------------------------------------------
    # HALAMAN 2: EKSPLORASI DATA MENTAH
    # ---------------------------------------------------------
    elif pathname == '/manage':
        # (Kode isi page_manage biarkan UTUH sama persis seperti kode aslimu sebelumnya)
        page_manage = html.Div([
            html.Div([
                html.Div("📊 Database Pembangunan Jatim", className="sec-title", style={"borderBottom": "none", "marginBottom": "0"}),
                html.Div("Sumber: BPS Jawa Timur (Data Asli)", style={"fontSize": "13px", "color": "#64748b", "fontWeight": "500"})
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}),
            
            html.Div([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Label("🔍 Cari Daerah / Kawasan", style={"fontSize": "12px", "fontWeight": "600", "color": "#64748b", "marginBottom": "8px"}),
                            dbc.Input(id='search-manage-text', placeholder="Ketik nama kota atau kawasan...", style={"borderRadius": "8px"})
                        ], width=4),
                        dbc.Col([
                            html.Label("📊 Pilih Indikator", style={"fontSize": "12px", "fontWeight": "600", "color": "#64748b", "marginBottom": "8px"}),
                            dcc.Dropdown(
                                id='filter-manage-col',
                                options=[{'label': 'IPM', 'value': 'IPM'}, {'label': 'PDRB Per Kapita', 'value': 'PDRB_Per_Kapita'}, {'label': 'Persentase Miskin', 'value': 'Persentase_Miskin'}, {'label': 'Gini Ratio', 'value': 'Gini_Ratio'}, {'label': 'Tingkat Pengangguran (TPT)', 'value': 'TPT'}],
                                placeholder="Pilih kolom...", style={"borderRadius": "8px"}
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("⚖️ Kondisi", style={"fontSize": "12px", "fontWeight": "600", "color": "#64748b", "marginBottom": "8px"}),
                            dcc.Dropdown(
                                id='filter-manage-op',
                                options=[{'label': 'Lebih Besar (>=)', 'value': '>='}, {'label': 'Lebih Kecil (<=)', 'value': '<='}],
                                value='>=', clearable=False, style={"borderRadius": "8px"}
                            )
                        ], width=2),
                        dbc.Col([
                            html.Label("🔢 Masukkan Angka", style={"fontSize": "12px", "fontWeight": "600", "color": "#64748b", "marginBottom": "8px"}),
                            dbc.Input(id='filter-manage-val', type="number", placeholder="Contoh: 10.5", style={"borderRadius": "8px"})
                        ], width=3),
                    ])
                ], style={"backgroundColor": "#f8fafc", "padding": "18px 20px", "borderRadius": "12px", "border": "1px solid #e2e8f0", "marginBottom": "20px"}),
                
                dash_table.DataTable(
                    id='table-manage-data', 
                    data=df_master.to_dict('records') if not df_master.empty else [],
                    columns=[{'name': i, 'id': i} for i in df_master.columns],
                    page_size=15, sort_action="native",
                    style_table={'overflowX': 'auto', 'borderRadius': '10px', 'border': '1px solid #e2e8f0'},
                    style_header={'backgroundColor': '#f8fafc', 'fontWeight': 'bold', 'color': '#1e293b', 'padding': '12px'},
                    style_cell={'textAlign': 'left', 'padding': '12px', 'fontFamily': 'Poppins', 'fontSize': '13px'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}]
                )
            ], className="card-custom")
        ])
        return page_manage, inactive_style, active_style, inactive_style

    # ---------------------------------------------------------
    # HALAMAN 3: DASHBOARD UTAMA (DEFAULT)
    # ---------------------------------------------------------
    else:
        page_dashboard = html.Div([
            filter_panel,
            tabs_menu,
            html.Div(id='main-content-container')
        ])
        return page_dashboard, active_style, inactive_style, inactive_style

# ---------------------------------------------------------
# CALLBACK: TOGGLE LIGHT / DARK MODE (UI Switcher)
# ---------------------------------------------------------
@app.callback(
    Output('theme-icon', 'children'),
    Output('theme-text', 'children'),
    Output('theme-store', 'data'),
    Output('theme-toggle-btn', 'style'),
    Output('app-wrapper', 'className'), # <--- OUTPUT BARU!
    Input('theme-toggle-btn', 'n_clicks'),
    Input('theme-store', 'data'),
    prevent_initial_call=True
)
def toggle_theme(n_clicks, current_theme):
    base_style = {"display": "flex", "alignItems": "center", "cursor": "pointer", "padding": "10px 15px", "borderRadius": "12px", "transition": "all 0.3s ease"}
    
    if current_theme == 'light':
        # Nyalakan Dark Mode
        icon = html.I(className="fa-solid fa-moon fa-lg", style={"color": "#818cf8"}) 
        text = "Dark Mode"
        btn_style = {**base_style, "backgroundColor": "#1e293b", "border": "1px solid #334155", "color": "white"}
        return icon, text, 'dark', btn_style, "dark-mode"
    else:
        # Nyalakan Light Mode
        icon = html.I(className="fa-solid fa-sun fa-lg", style={"color": "#f59e0b"}) 
        text = "Light Mode"
        btn_style = {**base_style, "backgroundColor": "#f8fafc", "border": "1px solid #e2e8f0"}
        return icon, text, 'light', btn_style, ""
    
# Callback untuk Update Dropdown Kabupaten berdasarkan Kawasan
@app.callback(
    Output('filter-kab', 'options'),
    Output('filter-kab', 'value'),
    Input('filter-kawasan', 'value')
)
def update_kab_dropdown(sel_kawasan):
    if df_master.empty:
        return [], []
    if sel_kawasan == "Semua Kawasan":
        kabs = sorted(df_master['Kabupaten_Kota'].dropna().unique().tolist())
    else:
        kabs = sorted(df_master[df_master['Kawasan'] == sel_kawasan]['Kabupaten_Kota'].dropna().unique().tolist())
    return [{'label': k, 'value': k} for k in kabs], kabs

# ─────────────────────────────────────────────
# 4. CALLBACK UTAMA (Render Isi Tab & Tab 1, 3)
# ─────────────────────────────────────────────
@app.callback(
    Output('main-content-container', 'children'),
    [Input('tabs-selector', 'value'),
     Input('filter-tahun', 'value'),
     Input('filter-kab', 'value'),
     Input('theme-store', 'data')] # <--- TAMBAH INI
)
def render_content(tab, sel_year, sel_kab, theme): # <--- TAMBAH 'theme'
    # Taruh ini di baris pertama setelah def:
    PLOTLY_TEMPLATE = "plotly_dark" if theme == 'dark' else "plotly_white"
    
    if df_master.empty or not sel_kab:
        return html.Div("⚠️ Silakan pilih minimal satu Kabupaten/Kota di filter atas.", className="warn-box")

    df_filtered = df_master[(df_master['Tahun'] == sel_year) & (df_master['Kabupaten_Kota'].isin(sel_kab))]
    
    # ---------------------------------------------------------
    # TAB 1: EXECUTIVE SUMMARY (REVISI FINAL: OPSI 2 & VISUAL LENGKAP)
    # ---------------------------------------------------------
    if tab == 'tab-1':
        # --- OPSI 2: Mengambil Angka Resmi Provinsi dari PROV_TREND ---
        if sel_year in PROV_TREND["years"]:
            idx_year = PROV_TREND["years"].index(sel_year)
            official_prov_ipm = PROV_TREND["ipm"][idx_year]
            kpi_sub_text = "Data Agregat Resmi BPS Jatim"
        else:
            official_prov_ipm = df_filtered['IPM'].mean()
            kpi_sub_text = "Rerata Aritmetik Wilayah"

        avg_miskin = df_filtered['Persentase_Miskin'].mean()
        kab_tertinggi = df_filtered.loc[df_filtered['IPM'].idxmax()]
        kab_terendah = df_filtered.loc[df_filtered['IPM'].idxmin()]
        
        PLOTLY_TEMPLATE = "plotly_dark" if theme == 'dark' else "plotly_white"
        
        # --- 1. Komparasi Capaian IPM (Bar Horizontal) ---
        df_bar = df_filtered.sort_values(by="IPM", ascending=True)
        fig_bar = px.bar(df_bar, x="IPM", y="Kabupaten_Kota", orientation='h', text="IPM", color="IPM", color_continuous_scale="Teal")
        
        # Garis Rerata Menggunakan Angka Resmi
        fig_bar.add_vline(x=official_prov_ipm, line_dash="dash", line_color="#ef4444", 
                          annotation_text=f"IPM Jatim: {official_prov_ipm:.2f}", annotation_position="bottom right")
        
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE, height=550, margin=dict(l=0, r=30, t=10, b=0), 
            showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Indeks Pembangunan Manusia", yaxis_title="",
            yaxis={'categoryorder':'total ascending', 'tickfont': {'size': 9}}
        )

        # --- 2. Analisis Konvergensi ---
        if sel_year == 2019:
            konvergensi_content = html.Div("Analisis konvergensi tersedia untuk periode pasca-tahun dasar 2019.", className="insight-box")
        else:
            df_2019 = df_master[(df_master['Tahun'] == 2019) & (df_master['Kabupaten_Kota'].isin(sel_kab))][['Kabupaten_Kota', 'IPM']].rename(columns={'IPM': 'IPM_2019'})
            df_curr = df_filtered[['Kabupaten_Kota', 'IPM']].rename(columns={'IPM': 'IPM_Current'})
            df_conv = pd.merge(df_curr, df_2019, on='Kabupaten_Kota')
            target_ipm = 85.0 
            df_conv['Indeks_Konvergensi'] = ((1 - ((target_ipm - df_conv['IPM_Current']) / (target_ipm - df_conv['IPM_2019']))) * 100).round(2)
            df_conv = df_conv.sort_values('Indeks_Konvergensi', ascending=True)
            
            fig_conv = px.bar(df_conv, x="Indeks_Konvergensi", y="Kabupaten_Kota", orientation='h', text="Indeks_Konvergensi", color="Indeks_Konvergensi", color_continuous_scale="algae")
            fig_conv.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_conv.update_layout(
                template=PLOTLY_TEMPLATE, height=550, margin=dict(l=0, r=40, t=10, b=0), 
                coloraxis_showscale=False, xaxis_title="Laju Konvergensi (%)", yaxis_title="", 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                yaxis={'tickfont': {'size': 9}}
            )
            konvergensi_content = dcc.Graph(figure=fig_conv, config={'displayModeBar': False})

        # --- 3. Dinamika Longitudinal ---
        df_prov = pd.DataFrame(PROV_TREND)
        fig_line = px.line(df_prov, x="years", y="ipm", markers=True)
        fig_line.update_traces(line_color="#0284c7", line_width=4, marker=dict(size=10, color="#059669"))
        fig_line.add_vline(x=2020, line_dash="dot", line_color="#e11d48", annotation_text="Krisis Pandemi", annotation_position="top right")
        fig_line.update_layout(template=PLOTLY_TEMPLATE, height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Periode", yaxis_title="Skor IPM", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        # --- 4. Evaluasi Disparitas (DENGAN LEGENDA & LABEL KEMBALI) ---
        fig_scatter = px.scatter(df_filtered, x="PDRB_Per_Kapita", y="IPM", size="Persentase_Miskin", color="Kawasan", 
                                 hover_name="Kabupaten_Kota", text="Kabupaten_Kota", 
                                 labels={"Kawasan": "Klasifikasi Wilayah"})
        fig_scatter.update_traces(textposition='top center', textfont=dict(size=9), marker=dict(sizemin=5))
        fig_scatter.update_layout(
            template=PLOTLY_TEMPLATE, height=400, margin=dict(l=0, r=0, t=10, b=0), 
            showlegend=True, # Legenda Kembali
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="PDRB per Kapita (Juta Rp)", yaxis_title="Skor IPM", 
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )

        return html.Div([
            html.Div("🎯 Indikator Kinerja Utama (IKU)", className="sec-title"),
            dbc.Row([
                dbc.Col(html.Div([html.Div("IPM Jawa Timur", className="kpi-title"), html.Div(f"{official_prov_ipm:.2f}", className="kpi-val", style={"color": "#0ea5e9"}), html.Div(kpi_sub_text, className="kpi-sub")], className="card-custom"), width=3),
                dbc.Col(html.Div([html.Div("Rerata Kemiskinan", className="kpi-title"), html.Div(f"{avg_miskin:.2f}%", className="kpi-val", style={"color": "#f43f5e"}), html.Div(f"Indikator Eksklusi Sosial", className="kpi-sub")], className="card-custom"), width=3),
                dbc.Col(html.Div([html.Div("Capaian Maksimum", className="kpi-title"), html.Div(f"{kab_tertinggi['IPM']:.2f}", className="kpi-val", style={"color": "#10b981"}), html.Div(f"Wilayah: {kab_tertinggi['Kabupaten_Kota']}", className="kpi-sub")], className="card-custom"), width=3),
                dbc.Col(html.Div([html.Div("Capaian Minimum", className="kpi-title"), html.Div(f"{kab_terendah['IPM']:.2f}", className="kpi-val", style={"color": "#f59e0b"}), html.Div(f"Wilayah: {kab_terendah['Kabupaten_Kota']}", className="kpi-sub")], className="card-custom"), width=3),
            ]),
            
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("📊 Komparasi Stratifikasi Capaian IPM Antar Wilayah", className="sec-title"),
                    dcc.Graph(figure=fig_bar, config={'displayModeBar': False}),
                    html.Div([html.B("Analisis Komparatif: "), "Lebarnya rentang skor IPM mengindikasikan signifikansi disparitas kualitas hidup antarwilayah. Hal ini mengisyaratkan perlunya afirmasi kebijakan alokasi belanja daerah untuk meminimalkan ketimpangan spasial terhadap pusat aglomerasi."], className="insight-box")
                ], className="card-custom"), width=6),

                dbc.Col(html.Div([
                    html.Div("⭐ Analisis Konvergensi: Laju Reduksi Kesenjangan", className="sec-title"),
                    konvergensi_content,
                    html.Div([html.B("Analisis Dinamis: "), "Pendekatan konvergensi mengukur laju suatu wilayah dalam mereduksi kesenjangan menuju ambang batas ideal (skor IPM 85) dengan referensi tahun dasar 2019. Capaian konvergensi yang rendah pada wilayah maju dapat menjadi indikator awal terjadinya saturasi pertumbuhan."], className="insight-box")
                ], className="card-custom"), width=6),
            ]),
            
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("📈 Dinamika Longitudinal Capaian IPM Provinsi", className="sec-title"), 
                    dcc.Graph(figure=fig_line, config={'displayModeBar': False}),
                    html.Div([html.B("Analisis Tren: "), "Kurva runtun waktu mengonfirmasi resiliensi struktural pembangunan manusia di Jawa Timur. Walaupun sempat terkontraksi akibat krisis pandemi, tren pertumbuhan kembali terakselerasi sejalan dengan stabilisasi makroekonomi daerah."], className="insight-box")
                ], className="card-custom"), width=5),
                
                dbc.Col(html.Div([
                    html.Div("💡 Evaluasi Disparitas Output Ekonomi & Kualitas Manusia", className="sec-title"), 
                    dcc.Graph(figure=fig_scatter, config={'displayModeBar': False}),
                    html.Div([html.B("Interpretasi Disparitas: "), "Wilayah dengan PDRB tinggi namun IPM rendah merepresentasikan pola pertumbuhan ekonomi yang kurang inklusif. Fenomena ini mengindikasikan bahwa akumulasi kapital belum bertransmisi secara optimal terhadap peningkatan kualitas pelayanan publik."], className="insight-box")
                ], className="card-custom"), width=7),
            ]),
        ])

    # ---------------------------------------------------------
    # TAB 2: DEKOMPOSISI IPM
    # ---------------------------------------------------------
    elif tab == 'tab-2':
        return html.Div([
            html.Div("🔍 Dekomposisi IPM & Analisis Korelasi Sektoral", className="sec-title"),
            
            html.Div([
                html.B("Rasionalisasi Analisis: "),
                "IPM merupakan indeks komposit. Halaman ini bertujuan untuk mendekomposisi (mengurai) IPM menjadi indikator penyusun dasarnya. Melalui analisis korelasi silang (cross-sectoral), evaluasi difokuskan untuk mengidentifikasi apakah kemajuan di satu sektor berjalan sinkron dengan sektor lainnya, atau justru mengindikasikan adanya ketimpangan prioritas pembangunan antarwilayah."
            ], className="insight-box", style={"marginBottom": "20px"}),

            # --- KOTAK DROPDOWN (UI FIX: zIndex 999 & margin-bottom) ---
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Variabel Sumbu X (Independen):", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Dropdown(id='tab2-x', options=[{'label': k, 'value': v} for k, v in opsi_indikator.items()], value="RLS", clearable=False, style={"borderRadius": "8px"})
                    ], width=6),
                    dbc.Col([
                        html.Label("Variabel Sumbu Y (Dependen):", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Dropdown(id='tab2-y', options=[{'label': k, 'value': v} for k, v in opsi_indikator.items()], value="UHH", clearable=False, style={"borderRadius": "8px"})
                    ], width=6)
                ]),
            ], className="card-custom", style={"padding": "15px 24px", "marginBottom": "25px", "position": "relative", "zIndex": "999"}), 
            
            # Wadah Kosong untuk Grafik
            html.Div(id='tab2-output-container')
        ])

    # ---------------------------------------------------------
    # TAB 3: EVALUASI INKLUSIVITAS EKONOMI
    # ---------------------------------------------------------
    elif tab == 'tab-3':
        cols_tabel = ['Kabupaten_Kota', 'Kawasan', 'IPM', 'PDRB_Per_Kapita', 'Persentase_Miskin', 'Gini_Ratio', 'TPT']
        if 'Garis_Kemiskinan' in df_filtered.columns:
            cols_tabel.append('Garis_Kemiskinan')
            
        # Urutkan dan Reset Index agar sesuai dengan baris tabel
        df_tabel = df_filtered[cols_tabel].sort_values(by='PDRB_Per_Kapita', ascending=False).reset_index(drop=True)
        
        # --- RUMUS PEMBUAT BACKGROUND GRADIENT ---
        def get_row_colors(df, col, color_type):
            styles = []
            if col not in df.columns: return styles
            cmin, cmax = df[col].min(), df[col].max()
            if cmax == cmin: return styles
            
            for i, val in enumerate(df[col]):
                if pd.isna(val): continue
                normalized = (val - cmin) / (cmax - cmin)
                
                # Setup warna RGB
                if color_type == 'blue': r, g, b = 2, 132, 199
                elif color_type == 'green': r, g, b = 5, 150, 105
                else: r, g, b = 225, 29, 72 # Merah
                
                alpha = 0.1 + (normalized * 0.8) 
                text_color = "white" if normalized > 0.6 else "black"
                
                styles.append({
                    'if': {'row_index': i, 'column_id': col},
                    'backgroundColor': f"rgba({r},{g},{b},{alpha})",
                    'color': text_color
                })
            return styles
            
        color_styles = []
        color_styles += get_row_colors(df_tabel, 'IPM', 'blue')
        color_styles += get_row_colors(df_tabel, 'PDRB_Per_Kapita', 'green')
        color_styles += get_row_colors(df_tabel, 'Persentase_Miskin', 'red')
        color_styles += get_row_colors(df_tabel, 'Gini_Ratio', 'red')
        color_styles += get_row_colors(df_tabel, 'TPT', 'red')
        
        # --- FORMATTING TEXT TABEL ---
        df_display = df_tabel.copy()
        df_display['Persentase_Miskin'] = df_display['Persentase_Miskin'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        df_display['TPT'] = df_display['TPT'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        df_display['Gini_Ratio'] = df_display['Gini_Ratio'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
        if 'Garis_Kemiskinan' in df_display.columns:
            df_display['Garis_Kemiskinan'] = df_display['Garis_Kemiskinan'].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "")
        
        # Heatmap Korelasi
        cols_corr = ['IPM', 'PDRB_Per_Kapita', 'Persentase_Miskin', 'Gini_Ratio', 'TPT']
        if 'Pengeluaran_Per_Kapita' in df_filtered.columns: cols_corr.append('Pengeluaran_Per_Kapita')
        
        corr_matrix = df_filtered[cols_corr].corr().round(2)
        fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='Tealrose', zmin=-1, zmax=1)
        fig_heat.update_layout(template=PLOTLY_TEMPLATE, height=400, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        daerah_terkaya = df_tabel.iloc[0] 
        nama_daerah = daerah_terkaya['Kabupaten_Kota']
        pdrb_val = daerah_terkaya['PDRB_Per_Kapita']
        gini_val = daerah_terkaya['Gini_Ratio']
        miskin_val = daerah_terkaya['Persentase_Miskin']
        
        if miskin_val > 10.0 or gini_val > 0.350:
            status_warna, status_teks, class_box = "#e11d48", "⚠️ Indikasi Pertumbuhan Eksklusif (Timpang)", "danger-box"
            kesimpulan = f"Kendati menduduki peringkat agregat ekonomi (PDRB) tertinggi, tingginya angka kemiskinan dan ketimpangan mengindikasikan bahwa akumulasi kapital di {nama_daerah} berpotensi hanya terkonsentrasi pada sektor padat modal tanpa menciptakan efek distribusi kesejahteraan (trickle-down effect) yang masif bagi masyarakat marginal."
        else:
            status_warna, status_teks, class_box = "#059669", "✅ Indikasi Pertumbuhan Inklusif", "success-box"
            kesimpulan = f"Analisis mengonfirmasi bahwa {nama_daerah} mampu mengonversi tingginya aktivitas makroekonomi menjadi peningkatan pemerataan sosial. Intervensi kebijakan terindikasi berhasil memitigasi ketimpangan pendapatan dan menekan angka kerentanan struktural."

        return html.Div([
            html.Div("⚖️ Evaluasi Inklusivitas Ekonomi & Distribusi Kesejahteraan", className="sec-title"),
            
            # --- RASIONALISASI ANALISIS (DI ATAS SEPERTI TAB 2) ---
            html.Div([
                html.B("Rasionalisasi Analisis: "),
                "Halaman ini menyajikan evaluasi inklusivitas pertumbuhan ekonomi melalui analisis keterkaitan antara output ekonomi dengan indikator kesejahteraan. Fokus utama adalah mengukur efektivitas transmisi pertumbuhan terhadap pengurangan kemiskinan dan pengangguran serta pemerataan pendapatan di tingkat wilayah."
            ], className="insight-box", style={"marginBottom": "20px"}),
            
            # Insight Box Akademis untuk Tabel
            html.Div([
                html.B("Metodologi Pemetaan Visual:"), html.Br(),
                "Tabel ini mengaplikasikan teknik pemetaan gradien warna untuk mengidentifikasi konsentrasi dan disparitas indikator lintas wilayah.", html.Br(),
                html.Ul([
                    html.Li([html.Span("🟩 Gradien Hijau (Kapasitas Fiskal/PDRB):", style={"fontWeight": "bold", "color": "#059669"}), " Merepresentasikan skala output makroekonomi wilayah."]),
                    html.Li([html.Span("🟥 Gradien Merah (Indikator Kerentanan):", style={"fontWeight": "bold", "color": "#e11d48"}), " Semakin pekat warna merah, semakin tinggi tingkat urgensi pada indikator kemiskinan, ketimpangan (Gini), maupun pengangguran (TPT)."]),
                    html.Li([html.Span("🟦 Gradien Biru (Kualitas Modal Manusia):", style={"fontWeight": "bold", "color": "#0284c7"}), " Merepresentasikan tingkat kemajuan indeks komposit IPM."])
                ], style={"marginTop": "8px", "marginBottom": "0"})
            ], className="insight-box", style={"marginBottom": "25px"}),
            
            html.Div([
                html.B(f"Matriks Indikator Kesejahteraan Multidimensi ({sel_year})", style={"fontSize": "15px", "display": "block", "marginBottom": "15px"}),
                dash_table.DataTable(
                    data=df_display.to_dict('records'), 
                    columns=[{'name': i, 'id': i} for i in df_display.columns],
                    cell_selectable=False, # <--- BUG FIX: BIKIN TABEL GAK MERAH PAS DIKLIK
                    style_table={'overflowX': 'auto', 'borderRadius': '10px', 'border': '1px solid #e2e8f0'},
                    style_header={'backgroundColor': '#f8fafc', 'fontWeight': 'bold', 'color': '#1e293b', 'padding': '12px'},
                    style_cell={'textAlign': 'left', 'padding': '12px', 'fontFamily': 'Poppins', 'borderBottom': '1px solid #e2e8f0'},
                    page_size=15, 
                    style_data_conditional=color_styles 
                )
            ], className="card-custom"),
            
                dbc.Row([
                dbc.Col(html.Div([
                    html.Div("🔥 Matriks Korelasi Kesejahteraan Multidimensi", className="sec-title", style={"fontSize": "15px"}),
                    dcc.Graph(figure=fig_heat, config={'displayModeBar': False}),
                    
                    html.Div([
                        html.B("Interpretasi Matriks: "), 
                        "Matriks korelasi ini memetakan kekuatan hubungan antarindikator makro menggunakan koefisien korelasi linear. Nilai positif menunjukkan hubungan searah antarvariabel, sedangkan nilai negatif menunjukkan hubungan berlawanan arah yang mencerminkan adanya perbedaan tren perkembangan antarindikator."
                    ], className="insight-box")
                ], className="card-custom"), width=7),
                
                dbc.Col(html.Div([
                    html.Div("🚨 Deteksi Anomali: Evaluasi Inklusivitas Pusat Pertumbuhan", className="sec-title", style={"fontSize": "15px"}),
                    html.Div([
                        html.B(f"Fokus Observasi: {nama_daerah} ({sel_year})"), html.Br(), html.Br(),
                        "Algoritma mengisolasi wilayah dengan PDRB per kapita absolut tertinggi guna mengevaluasi sejauh mana ekuilibrium (keseimbangan) distribusi kekayaan tercapai.", html.Br(), html.Br(),
                        html.Ul([
                            html.Li([html.B("Kapasitas Ekonomi: "), f"PDRB per Kapita tercatat sebesar {pdrb_val:.1f} Juta Rp."]),
                            html.Li([html.B("Metrik Distribusi: "), html.I("Gini Ratio "), f"{gini_val:.3f} | Tingkat Kemiskinan {miskin_val:.2f}%."]),
                            html.Li([html.B("Status Ekstraksi: "), html.Span(status_teks, style={"color": status_warna, "fontWeight": "bold"})])
                        ], style={"paddingLeft": "20px"}),
                        html.Br(), html.B("Konklusi Analitis: "), kesimpulan
                    ], className=class_box)
                ], className="card-custom"), width=5)
            ])
        ])

    # ---------------------------------------------------------
    # TAB 4: MACHINE LEARNING (Hanya Kerangka & Input)
    # ---------------------------------------------------------
    elif tab == 'tab-4':
        return html.Div([
            html.Div("🤖 Machine Learning: Klasterisasi & Evaluasi Model", className="sec-title"),
            html.Div([
                html.Label("🎛️ Atur K (Jumlah Kelompok):", style={"fontWeight": "600", "fontSize": "14px", "marginBottom": "10px"}),
                dcc.Slider(id='tab4-k-slider', min=2, max=5, step=1, value=3, marks={i: str(i) for i in range(2, 6)})
            ], className="card-custom", style={"padding": "20px 24px", "borderLeft": "4px solid #8b5cf6"}),
            
            # Wadah Kosong untuk Hasil ML Tab 4 (Akan diisi oleh callback ke-3)
            html.Div(id='tab4-output-container')
        ])

# ─────────────────────────────────────────────
# 5. CALLBACK KHUSUS TAB 2 (Logika Plot & Insight)
# ─────────────────────────────────────────────
@app.callback(
    Output('tab2-output-container', 'children'),
    [Input('tab2-x', 'value'), Input('tab2-y', 'value'), Input('filter-tahun', 'value'), Input('filter-kab', 'value'), Input('theme-store', 'data')] # <--- TAMBAH
)
def render_tab2(var_x, var_y, sel_year, sel_kab, theme): # <--- TAMBAH
    PLOTLY_TEMPLATE = "plotly_dark" if theme == 'dark' else "plotly_white"
    
    if not var_x or not var_y or df_master.empty or not sel_kab: return dash.no_update
    df_filtered = df_master[(df_master['Tahun'] == sel_year) & (df_master['Kabupaten_Kota'].isin(sel_kab))]
    
    # Buat dictionary terbalik untuk mendapatkan nama label
    inv_opsi = {v: k for k, v in opsi_indikator.items()}
    var_x_label, var_y_label = inv_opsi.get(var_x, var_x), inv_opsi.get(var_y, var_y)

    fig_dekon = px.scatter(df_filtered, x=var_x, y=var_y, color="Kawasan", text="Kabupaten_Kota", size="IPM", hover_name="Kabupaten_Kota", template=PLOTLY_TEMPLATE, color_discrete_sequence=px.colors.qualitative.Safe)
    fig_dekon.update_layout(height=450, margin=dict(l=0, r=40, t=40, b=0), legend=dict(orientation="h", y=1.1), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    korelasi = df_filtered[var_x].corr(df_filtered[var_y])
    kab_max_x = df_filtered.loc[df_filtered[var_x].idxmax(), 'Kabupaten_Kota']
    kab_max_y = df_filtered.loc[df_filtered[var_y].idxmax(), 'Kabupaten_Kota']

    if korelasi > 0.7:
        kategori_kor, interpretasi, rekomendasi = "Positif Signifikan (Kuat)", "Integrasi sektoral berjalan sangat selaras. Kemajuan pada indikator X secara konsisten ekuivalen dengan peningkatan indikator Y di mayoritas wilayah.", "Pertahankan integrasi kebijakan lintas sektor. Afirmasi anggaran direkomendasikan bagi wilayah yang posisinya berada jauh di bawah garis regresi linear."
    elif korelasi > 0.3:
        kategori_kor, interpretasi, rekomendasi = "Positif Moderat", "Terdapat kecenderungan korelasi searah yang moderat. Hal ini mengindikasikan adanya variasi disparitas, di mana beberapa wilayah unggul di satu sektor namun tertinggal di sektor lain.", "Dibutuhkan pendekatan intervensi spesifik (targeted intervention). Prioritaskan investigasi pada wilayah 'outlier' yang memiliki asimetri skor antara sumbu X dan Y."
    elif korelasi < -0.3:
        kategori_kor, interpretasi, rekomendasi = "Negatif (Trade-Off)", "Terdeteksi anomali struktural. Peningkatan satu sektor diiringi oleh penurunan sektor lain, mengindikasikan potensi ketimpangan alokasi sumber daya.", "Diperlukan evaluasi fiskal komprehensif. Pastikan tidak ada kebijakan trade-off yang mengorbankan pendanaan sektor Y demi akselerasi sektor X."
    else:
        kategori_kor, interpretasi, rekomendasi = "Lemah / Tidak Signifikan", "Tidak teridentifikasi pola korelasi linear. Kedua indikator berkembang secara independen tanpa efek rambat (spillover effect) yang nyata.", "Evaluasi program wajib dilakukan secara spesifik per sektor. Peningkatan target di satu indikator tidak akan secara otomatis mengeskalasi capaian di indikator lainnya."

    teks_sorotan = f"Wilayah {kab_max_x} mendominasi capaian tertinggi pada kedua indikator secara simultan." if kab_max_x == kab_max_y else f"Sebagai perbandingan komparatif, {kab_max_x} mencatat rekor tertinggi di sumbu X ({var_x_label}), sedangkan {kab_max_y} memimpin di sumbu Y ({var_y_label})."

    # Penanganan khusus jika User memilih filter Madura atau GKS
    sel_kawasan = df_master[df_master['Kabupaten_Kota'].isin(sel_kab)]['Kawasan'].mode()[0] if len(sel_kab) > 0 else ""
    catatan_khusus = ""
    if "Madura" in sel_kawasan and len(sel_kab) <= 4:
        catatan_khusus = html.Div(html.I("Catatan Khusus Madura: Secara geografis, wilayah ini (terutama Bangkalan) terhubung langsung dengan pusat fasilitas di Surabaya. Namun, karena capaiannya sering berada di klaster terbawah, ini membuktikan bahwa jarak fisik bukanlah hambatan utama, melainkan butuh intervensi akses sosiokultural."), style={"marginTop": "15px", "paddingTop": "15px", "borderTop": "1px dashed #cbd5e1"})
    elif "GKS" in sel_kawasan and len(sel_kab) <= 5:
        catatan_khusus = html.Div(html.I("Catatan Khusus GKS: Visualisasi ini menyoroti dengan jelas ketimpangan (gap) pembangunan yang cukup lebar antara Surabaya dan Sidoarjo sebagai aglomerasi utama, dibandingkan wilayah penyangga lainnya."), style={"marginTop": "15px", "paddingTop": "15px", "borderTop": "1px dashed #cbd5e1"})

    return html.Div([
        html.Div(dcc.Graph(figure=fig_dekon, config={'displayModeBar': False}), className="card-custom"),
        
        dbc.Row([
            dbc.Col(html.Div([
                html.B(f"💡 Analisis Cerdas ({sel_year}):", style={"color": "#059669", "fontSize": "16px", "marginBottom": "10px", "display": "block"}),
                html.Ul([
                    html.Li([html.B("Skor Korelasi: "), html.Span(f"{korelasi:.2f}", style={"color": "#0284c7", "fontWeight": "bold"}), f" ({kategori_kor})."], style={"marginBottom": "8px"}),
                    html.Li([html.B("Interpretasi: "), interpretasi], style={"marginBottom": "8px"}),
                    html.Li([html.B("Sorotan Daerah: "), teks_sorotan], style={"marginBottom": "8px"}),
                    html.Li([html.B("Saran Kebijakan: "), rekomendasi], style={"marginBottom": "0"})
                ]),
                catatan_khusus
            ], className="insight-box"), width=7),
            
            dbc.Col(html.Div([
                html.B("📊 Edukasi Metodologi: Membaca Skor Korelasi Pearson", style={"fontSize": "14px"}), html.Br(), html.Br(),
                html.B("Panduan Membaca Skor (r):"), html.Br(),
                html.Code("0.70 s/d  1.00 : Sangat Kuat"), html.Br(),
                html.Code("0.40 s/d  0.69 : Kuat/Moderat"), html.Br(),
                html.Code("-1.00 s/d -0.10 : Negatif (Berlawanan)"), html.Br(), html.Br(),
                "Analisis ini menggunakan Koefisien Korelasi Pearson untuk mengukur arah dan kekuatan hubungan linear.", html.Br(),
                html.I("Catatan: Korelasi mengukur hubungan, namun bukan berarti satu hal secara pasti menyebabkan hal lainnya (Correlation does not imply causation).")
            ], className="info-box", style={"background": "#f8fafc", "padding": "20px", "borderRadius": "12px", "border": "1px solid #e2e8f0", "marginTop": "15px", "fontSize": "13px"}), width=5)
        ])
    ])

# ---------------------------------------------------------
# 6. CALLBACK ANALISIS KLASTER (ACADEMIC VERSION)
# ---------------------------------------------------------
@app.callback(
    Output('tab4-output-container', 'children'),
    [Input('tab4-k-slider', 'value'), 
     Input('filter-tahun', 'value'), 
     Input('filter-kab', 'value'), 
     Input('theme-store', 'data')]
)
def render_tab4(num_clusters, sel_year, sel_kab, theme):
    PLOTLY_TEMPLATE = "plotly_dark" if theme == 'dark' else "plotly_white"
    # Pengaturan warna dinamis untuk memperbaiki bug Dark Mode
    bg_color = "#1e293b" if theme == 'dark' else "#f8fafc"
    border_color = "#334155" if theme == 'dark' else "#e2e8f0"
    text_color = "#f8fafc" if theme == 'dark' else "#1e293b"
    
    if not num_clusters or df_master.empty or not sel_kab: return dash.no_update
    df_filtered = df_master[(df_master['Tahun'] == sel_year) & (df_master['Kabupaten_Kota'].isin(sel_kab))]
    
    if len(df_filtered) < 5:
        return html.Div("Peringatan: Diperlukan minimal 5 observasi wilayah agar algoritma pengelompokan dapat beroperasi secara optimal.", className="warn-box")

    # Proses K-Means
    fitur_ml = ['IPM', 'PDRB_Per_Kapita', 'Persentase_Miskin', 'RLS']
    df_ml = df_filtered[['Kabupaten_Kota'] + fitur_ml].copy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_ml[fitur_ml])
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_ml['Cluster_Raw'] = kmeans.fit_predict(data_scaled)
    
    idx_urut = df_ml.groupby('Cluster_Raw')['IPM'].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(idx_urut)}
    df_ml['Cluster_Urut'] = df_ml['Cluster_Raw'].map(mapping)
    df_ml['Nama_Cluster'] = "Klaster " + (df_ml['Cluster_Urut'] + 1).astype(str)
    
    df_plot = pd.merge(df_filtered, df_ml[['Kabupaten_Kota', 'Nama_Cluster']], on='Kabupaten_Kota')
    score = silhouette_score(data_scaled, df_ml['Cluster_Raw'])

    # Evaluasi Akademik Silhouette Score
    if score > 0.5: 
        msg, warna = "Struktur Klaster Terdefinisi dengan Baik (Strong)", "#059669"
    elif score > 0.25: 
        msg, warna = "Struktur Klaster Terklasifikasi Cukup (Moderate)", "#d97706"
    else: 
        msg, warna = "Struktur Klaster Lemah (Data Tumpang Tindih)", "#e11d48"

    fig_3d = px.scatter_3d(df_plot, x='PDRB_Per_Kapita', y='Persentase_Miskin', z='IPM', color='Nama_Cluster', text='Kabupaten_Kota', color_discrete_sequence=px.colors.qualitative.Prism)
    fig_3d.update_layout(template=PLOTLY_TEMPLATE, height=500, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    profil = df_plot.groupby('Nama_Cluster')[fitur_ml].mean().round(2).reset_index()

    cluster_summaries = []
    for i in range(num_clusters):
        c_name = f"Klaster {i+1}"
        members = df_plot[df_plot['Nama_Cluster'] == c_name]['Kabupaten_Kota'].tolist()
        m_list = ", ".join(members)
        
        # Penamaan klaster yang lebih akademis
        if i == 0: 
            tag, desc = "Prioritas Pengembangan", "Kelompok wilayah dengan karakteristik indeks pembangunan rendah dan tingkat kerentanan ekonomi tinggi. Wilayah ini memerlukan intervensi kebijakan afirmatif."
        elif i == num_clusters - 1: 
            tag, desc = "Mandiri dan Terakselerasi", "Kelompok wilayah dengan kapasitas ekonomi makro yang dominan serta kualitas modal manusia yang telah melampaui rata-rata provinsi."
        else: 
            tag, desc = "Transisional", "Kelompok wilayah dalam tahap perkembangan menengah yang memerlukan optimalisasi pada sektor jasa dan produktivitas tenaga kerja."
        
        cluster_summaries.append(html.Div([
            html.B(f"{c_name}: {tag}"), html.Br(),
            html.Span("Anggota Wilayah: ", style={"color": "#0284c7", "fontWeight": "bold"}), html.I(m_list), html.Br(),
            html.Span("Analisis Karakteristik: ", style={"color": "#059669", "fontWeight": "bold"}), desc
        ], style={"marginBottom": "15px", "paddingBottom": "15px", "borderBottom": f"1px solid {border_color}", "color": text_color}))

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("Skor Kohesi Klaster (Silhouette)", className="kpi-title"),
                html.Div(f"{score:.3f}", className="kpi-val", style={"color": "#0284c7"}),
                html.Div(msg, className="kpi-sub", style={"color": warna, "fontWeight": "bold", "marginTop": "5px"}),
            ], className="card-custom"), width=4),
            
            dbc.Col(html.Div([
                html.B("Evaluasi Metodologi: Validasi Silhouette Score", style={"fontSize": "14px"}), html.Br(),
                "Koefisien Silhouette mengukur derajat kemiripan suatu objek dengan klasternya sendiri dibandingkan dengan klaster lain. Skor yang mendekati nilai 1.0 mengindikasikan bahwa proses segmentasi wilayah telah mencapai tingkat ekuilibrium yang optimal.",
            ], className="info-box", style={"background": bg_color, "color": text_color, "padding": "20px", "borderRadius": "12px", "border": f"1px solid {border_color}", "fontSize": "13px"}), width=8)
        ]),
        
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("🌌 Pemetaan Klasterisasi Spasial 3D", className="sec-title", style={"fontSize": "15px"}),
                dcc.Graph(figure=fig_3d, config={'displayModeBar': False}),
                html.Div("Analisis Spasial: Posisi kedekatan antar titik merepresentasikan kemiripan profil sosio-ekonomi wilayah berdasarkan variabel IPM, PDRB, dan Kemiskinan.", className="insight-box")
            ], className="card-custom"), width=8),
            
            dbc.Col(html.Div([
                html.Div("📊 Profil Rerata Indikator per Klaster", className="sec-title", style={"fontSize": "15px"}),
                dash_table.DataTable(
                    data=profil.to_dict('records'), columns=[{'name': i, 'id': i} for i in profil.columns],
                    style_table={'overflowX': 'auto', 'borderRadius': '8px', 'border': f"1px solid {border_color}"},
                    style_header={'backgroundColor': bg_color, 'fontWeight': 'bold', 'color': text_color},
                    style_cell={'backgroundColor': bg_color, 'color': text_color, 'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                ),
                html.Div([
                    html.B("Panduan Komparasi:"), html.Br(),
                    "Perbedaan nilai rerata antarbaris menunjukkan spesialisasi masalah di tiap klaster. Klaster dengan nilai IPM terendah direkomendasikan menjadi fokus program pengentasan kemiskinan struktural."
                ], style={"marginTop": "15px", "fontSize": "12.5px", "color": "#64748b" if theme == 'light' else "#94a3b8"})
            ], className="card-custom"), width=4)
        ]),
        
        html.Div([
            html.Div("🧠 Analisis Karakteristik Klaster Wilayah", className="sec-title", style={"fontSize": "15px"}),
            html.Div(cluster_summaries, style={"fontSize": "13.5px"})
        ], className="card-custom")
    ])

# ---------------------------------------------------------
# CALLBACK KHUSUS HALAMAN MANAGE DATA (FILTER DINAMIS)
# ---------------------------------------------------------
@app.callback(
    Output('table-manage-data', 'data'),
    [Input('search-manage-text', 'value'),
     Input('filter-manage-col', 'value'),
     Input('filter-manage-op', 'value'),
     Input('filter-manage-val', 'value')]
)
def update_manage_table(search_text, filter_col, filter_op, filter_val):
    if df_master.empty: 
        return []

    dff = df_master.copy()

    # 1. Filter Pencarian Teks
    if search_text:
        search_text = search_text.lower()
        mask = dff['Kabupaten_Kota'].str.lower().str.contains(search_text, na=False) | \
               dff['Kawasan'].str.lower().str.contains(search_text, na=False)
        dff = dff[mask]

    # 2. Filter Numerik Dinamis (Berdasarkan Pilihan User)
    if filter_col and filter_op and filter_val is not None:
        if filter_op == '>=':
            dff = dff[dff[filter_col] >= filter_val]
        elif filter_op == '<=':
            dff = dff[dff[filter_col] <= filter_val]

    return dff.to_dict('records')

# ─────────────────────────────────────────────
# 7. RUN SERVER
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)