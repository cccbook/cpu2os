#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <curl/curl.h>

 struct memory {
   char *response;
   size_t size;
 };
 
 static size_t cb(void *data, size_t size, size_t nmemb, void *userp)
 {
   size_t realsize = size * nmemb;
   struct memory *mem = (struct memory *)userp;
 
   char *ptr = realloc(mem->response, mem->size + realsize + 1);
   if(ptr == NULL)
     return 0;  /* out of memory! */
 
   mem->response = ptr;
   memcpy(&(mem->response[mem->size]), data, realsize);
   mem->size += realsize;
   mem->response[mem->size] = 0;
 
   return realsize;
 }
 
 struct memory chunk = {0};

int main(int argc, char *argv[]) {
    CURL *curl = curl_easy_init();
    if(!curl) return -1;

    curl_easy_setopt(curl, CURLOPT_URL, argv[1]);
    // If you are using libcurl as a win32 DLL, you MUST use a CURLOPT_WRITEFUNCTION if you set this option or you will experience crashes.
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);


    CURLcode res = curl_easy_perform(curl);

    if (res) return -1;

    curl_easy_cleanup(curl);

    printf("chunk.response=%s\n", chunk.response);
    return 0;
}

