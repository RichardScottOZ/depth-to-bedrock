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

#### saga_cmd
- Error in system(paste(shQuote(saga_cmd), "--version"), intern = T) : 
  'CreateProcess' failed to run 'C:\Users\rscott\DOWNLO~1\SAGA-9~1.2_X --version'
- you can pass in the actual binary to the function when testing - but full path  
- saga <- Rsagacmd::saga_gis(raster_format = "SAGA", all_outputs = FALSE, saga_bin='C:/Users/rscott/Downloads/saga-9.3.2_x64/saga_cmd.exe')

# Notes
- spatial raster temp in places like this
- C:\Users\rscott\AppData\Local\Temp\RtmpcHNmot

- C:\Users\rscott\AppData\Local\Temp\RtmpcHNmot

- MODIS files similarlry
  - MOD09A1.A2020185.h10v03.061.2020340144126.hdf



