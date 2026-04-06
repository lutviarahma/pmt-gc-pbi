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


def run_pipeline(tahun, mode, files, output, provinsi, chunk, custom_map):
    col = get_column_mapping(tahun, custom_map)
    if col is None:
        print(f"[ERROR] Tahun {tahun} tidak ada di mapping. Gunakan --map untuk custom mapping.")
        return

    print(f"\n{'='*50}")
    print(f"TAHUN {tahun} | SKENARIO: {mode.upper()}")
    print(f"{'='*50}")

    if mode == 'merged':
        preprocess(files['merged'], tahun, col, output, provinsi, chunk)

    elif mode == 'kor':
        raw_ruta_path = join_kor_kp(files['kor'], files['kp'], output_folder=output)
        if not raw_ruta_path:
            return
        if files.get('individu'):
            merged_path = merge_ruta_individu(
                ruta_path=raw_ruta_path,
                individu_path=files['individu'],
                output_folder=output,
                year=tahun
            )
            if not merged_path:
                return
        else:
            merged_path = raw_ruta_path
        preprocess(merged_path, tahun, col, output, provinsi, chunk)

    elif mode == 'ruta':
        merged_path = merge_ruta_individu(
            ruta_path=files['ruta'],
            individu_path=files['individu'],
            output_folder=output,
            year=tahun
        )
        if not merged_path:
            return
        preprocess(merged_path, tahun, col, output, provinsi, chunk)


def main():
    parser = argparse.ArgumentParser(
        description='Interface terpusat preprocessing Susenas PMT.\nUser tinggal masukkan data + tahun, script otomatis jalankan pipeline yang sesuai.',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--output', type=str, default='.', help='Folder output (default: direktori saat ini)')
    parser.add_argument('--chunk', type=int, default=100000, help='Chunk size untuk hemat RAM (default: 100000)')
    parser.add_argument('--map', type=str, nargs='*', default=[], metavar='KEY=VALUE',
                        help='Custom mapping kolom untuk tahun baru. Contoh: --map c_prov=PROP c_kab=KAB')

    sub = parser.add_subparsers(dest='command', required=True)

    # Skenario 1: KOR + KP terpisah
    p1 = sub.add_parser('kor', help='Skenario 1: KOR + KP terpisah -> join -> merge -> prepo')
    p1.add_argument('--tahun', type=int, required=True, nargs='+', help='Tahun data. Bisa lebih dari satu.')
    p1.add_argument('--kor', type=str, required=True, nargs='+', help='Path file KOR per tahun (urutan sesuai --tahun)')
    p1.add_argument('--kp', type=str, required=True, nargs='+', help='Path file KP per tahun (urutan sesuai --tahun)')
    p1.add_argument('--individu', type=str, nargs='+', default=[], help='Path file individu per tahun (opsional, urutan sesuai --tahun)')

    # Skenario 2: ruta + individu terpisah
    p2 = sub.add_parser('ruta', help='Skenario 2: ruta + individu terpisah -> merge -> prepo')
    p2.add_argument('--tahun', type=int, required=True, nargs='+', help='Tahun data. Bisa lebih dari satu.')
    p2.add_argument('--ruta', type=str, required=True, nargs='+', help='Path file ruta per tahun (urutan sesuai --tahun)')
    p2.add_argument('--individu', type=str, required=True, nargs='+', help='Path file individu per tahun (urutan sesuai --tahun)')

    # Skenario 3: sudah merged
    p3 = sub.add_parser('merged', help='Skenario 3: data sudah merged -> langsung prepo')
    p3.add_argument('--tahun', type=int, required=True, nargs='+', help='Tahun data. Bisa lebih dari satu.')
    p3.add_argument('--file', type=str, required=True, nargs='+', help='Path file merged per tahun (urutan sesuai --tahun)')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    custom_map = parse_custom_map(args.map) if args.map else None

    if args.command == 'merged':
        if len(args.tahun) != len(args.file):
            print("[ERROR] Jumlah --tahun dan --file harus sama.")
            sys.exit(1)
        for tahun, file_path in zip(args.tahun, args.file):
            run_pipeline(
                tahun=tahun,
                mode='merged',
                files={'merged': file_path},
                output=args.output,
                provinsi=[],
                chunk=args.chunk,
                custom_map=custom_map
            )

    elif args.command == 'kor':
        if len(args.tahun) != len(args.kor) or len(args.tahun) != len(args.kp):
            print("[ERROR] Jumlah --tahun, --kor, dan --kp harus sama.")
            sys.exit(1)
        individu_list = args.individu if args.individu else [None] * len(args.tahun)
        if args.individu and len(args.individu) != len(args.tahun):
            print("[ERROR] Jumlah --individu harus sama dengan --tahun jika diisi.")
            sys.exit(1)
        for i, tahun in enumerate(args.tahun):
            run_pipeline(
                tahun=tahun,
                mode='kor',
                files={
                    'kor': args.kor[i],
                    'kp': args.kp[i],
                    'individu': individu_list[i]
                },
                output=args.output,
                provinsi=[],
                chunk=args.chunk,
                custom_map=custom_map
            )

    elif args.command == 'ruta':
        if len(args.tahun) != len(args.ruta) or len(args.tahun) != len(args.individu):
            print("[ERROR] Jumlah --tahun, --ruta, dan --individu harus sama.")
            sys.exit(1)
        for i, tahun in enumerate(args.tahun):
            run_pipeline(
                tahun=tahun,
                mode='ruta',
                files={
                    'ruta': args.ruta[i],
                    'individu': args.individu[i]
                },
                output=args.output,
                provinsi=[],
                chunk=args.chunk,
                custom_map=custom_map
            )

    print("\nSemua proses selesai.")


if __name__ == '__main__':
    main()
