num_loops=20
per_loop=30000
hin_eng=false
while [$1 != ""]; do
    case $1 in
    "--num_workers")
        shift
        num_loops=$1
        ;;
    "--per_worker")
        shift
        per_loop=$1
        ;;
    "--hin_eng")
        hin_eng=true
        ;;
    esac
    shift
done
python data_gen.py --num_loops $num_loops --per_loop $per_loop --hin_eng $hin_eng
python skeletonize.py --num_loops $num_loops
python format_file_structure.py --num_loops $num_loops
rm -r ./dataset/o*
