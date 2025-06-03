
function clean_fl() {
    clFILE=$1
    rm ${clFILE} || true
}

function touch_fd() {
    tFOLD=$1
    mkdir -p ${tFOLD} || true
}

function clean_touch_fd() {
    ctFOLD=$1PB_FD
    rm -rf ${ctFOLD} || true
    mkdir -p ${ctFOLD}
}
