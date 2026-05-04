import argparse
import os
import sys

from preprocess import (
    join_kor_kp,
    merge_ruta_individu,
    preprocess,
    get_column_mapping,
    parse_custom_map,
)


def _banner(year):
    print(f"\n{'=' * 50}\nTAHUN {year}\n{'=' * 50}")


def cmd_merged(args):
    if len(args.tahun) != len(args.file):
        print("[ERROR] Jumlah --tahun dan --file harus sama.")
        sys.exit(1)
    custom_map = parse_custom_map(args.map) if args.map else None

    for y, f in zip(args.tahun, args.file):
        col = get_column_mapping(y, custom_map)
        if col is None:
            print(f"[SKIP] Tahun {y} tidak ada mapping.")
            continue
        _banner(y)
        if not os.path.exists(f):
            print(f"[ERROR] File tidak ditemukan: {f}")
            continue
        preprocess(f, y, col, args.output, args.provinsi, args.chunk)


def cmd_kor(args):
    custom_map = parse_custom_map(args.map) if args.map else None
    col = get_column_mapping(args.tahun, custom_map)
    if col is None:
        sys.exit(1)
    _banner(args.tahun)

    raw_ruta_path = join_kor_kp(args.kor, args.kp, args.output, year=args.tahun)
    if not raw_ruta_path:
        sys.exit(1)
    if args.individu:
        merged_path = merge_ruta_individu(raw_ruta_path, args.individu,
                                          args.output, args.tahun)
        if not merged_path:
            sys.exit(1)
    else:
        merged_path = raw_ruta_path
    preprocess(merged_path, args.tahun, col, args.output,
               args.provinsi, args.chunk)


def cmd_ruta(args):
    custom_map = parse_custom_map(args.map) if args.map else None
    col = get_column_mapping(args.tahun, custom_map)
    if col is None:
        sys.exit(1)
    _banner(args.tahun)

    merged_path = merge_ruta_individu(args.ruta, args.individu,
                                      args.output, args.tahun)
    if not merged_path:
        sys.exit(1)
    preprocess(merged_path, args.tahun, col, args.output,
               args.provinsi, args.chunk)


def _add_common(p):
    p.add_argument('--output', type=str, default='.',
                   help='Folder output (default: .)')
    p.add_argument('--provinsi', type=int, nargs='*', default=[],
                   help='Filter kode provinsi.')
    p.add_argument('--chunk', type=int, default=100000,
                   help='Chunk size untuk hemat RAM (default: 100000)')
    p.add_argument('--map', type=str, nargs='*', default=[],
                   metavar='KEY=VALUE',
                   help='Custom mapping kolom untuk tahun baru.')


def main():
    parser = argparse.ArgumentParser(
        description='Interface terpusat preprocessing Susenas PMT (3 skenario)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest='cmd', required=True,
                                metavar='{merged,kor,ruta}')

    pm = sub.add_parser('merged', help='Skenario 3: data sudah merged.')
    pm.add_argument('--tahun', type=int, nargs='+', required=True,
                    help='Boleh banyak tahun. Contoh: --tahun 2023 2024')
    pm.add_argument('--file', type=str, nargs='+', required=True,
                    help='File per tahun (urutan harus sesuai --tahun).')
    _add_common(pm)
    pm.set_defaults(func=cmd_merged)

    pk = sub.add_parser('kor', help='Skenario 1: KOR + KP terpisah '
                                    '(opsional + individu).')
    pk.add_argument('--tahun', type=int, required=True)
    pk.add_argument('--kor', type=str, required=True, help='File KOR.')
    pk.add_argument('--kp', type=str, required=True, help='File KP.')
    pk.add_argument('--individu', type=str, default=None,
                    help='File individu (opsional).')
    _add_common(pk)
    pk.set_defaults(func=cmd_kor)

    pr = sub.add_parser('ruta', help='Skenario 2: ruta + individu terpisah.')
    pr.add_argument('--tahun', type=int, required=True)
    pr.add_argument('--ruta', type=str, required=True)
    pr.add_argument('--individu', type=str, required=True)
    _add_common(pr)
    pr.set_defaults(func=cmd_ruta)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    args.func(args)
    print("\nSemua proses selesai.")


if __name__ == '__main__':
    main()
