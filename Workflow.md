# Errors
## Windows 10

Warning: CURL error: schannel: CertGetCertificateChain trust error CERT_TRUST_IS_UNTRUSTED_ROOT (GDAL error 11)Error: [rast] file does not exist: /vsicurl/https://ai4edataeuwest.blob.core.windows.net/alos-dem/AW3D30_global/ALPSMLC30_N061W122_DSM.tif

### Bypass certificate issue for now
- https://github.com/rspatial/terra/issues/608
- setGDALconfig(c("GDAL_HTTP_UNSAFESSL=YES"))

### vsicurl error
- get rid of /vsicurl/

#### Function to remove the /vsicurl/ prefix
```R
remove_vsicurl <- function(url) {
  gsub("^/vsicurl/", "", url)
}

# Apply the function to the URLs vector
new_urls <- lapply(urls, remove_vsicurl)
```
