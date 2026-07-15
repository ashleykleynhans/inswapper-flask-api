variable "REGISTRY" {
    default = "docker.io"
}

variable "REGISTRY_USER" {
    default = "ashleykza"
}

variable "APP" {
    default = "inswapper-flask-api"
}

variable "RELEASE" {
    default = "2.0.0"
}

variable "CU_VERSION" {
    default = "124"
}

variable "CUDA_VERSION" {
    default = "12.4.1"
}

variable "TORCH_VERSION" {
    default = "2.6.0"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["${REGISTRY}/${REGISTRY_USER}/${APP}:${RELEASE}"]
    annotations = [
        "org.opencontainers.image.title=${APP}",
        "org.opencontainers.image.description=FastAPI-powered face swapping API with 13 models, CodeFormer restoration, and VRAM-safe serial queue",
        "org.opencontainers.image.version=${RELEASE}",
        "org.opencontainers.image.vendor=ashleykleynhans",
        "org.opencontainers.image.source=https://github.com/ashleykleynhans/${APP}",
    ]
    attest = [
        "type=provenance,mode=max",
        "type=sbom",
    ]
    args = {
        CUDA_VERSION = "${CUDA_VERSION}"
        INDEX_URL = "https://download.pytorch.org/whl/cu${CU_VERSION}"
        TORCH_VERSION = "${TORCH_VERSION}+cu${CU_VERSION}"
    }
}
