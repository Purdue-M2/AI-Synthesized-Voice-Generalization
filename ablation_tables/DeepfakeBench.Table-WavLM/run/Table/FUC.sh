
function clean_fl() {
    local clFILE=$1
    rm ${clFILE} || true
}

function touch_fd() {
    local tFOLD=$1
    mkdir -p ${tFOLD} || true
}

function clean_touch_fd() {
    local ctFOLD=$1PB_FD
    rm -rf ${ctFOLD} || true
    mkdir -p ${ctFOLD}
}
