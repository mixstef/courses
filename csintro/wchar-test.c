#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <locale.h>

int main() {

/*
// testing a (multi)byte string
char *s = "123 δοκιμή";
int l = strlen(s);

printf("string is [%s], length=%d\n",s,l); // string displayed ok, length wrong
for (int i=0;i<l;i++) {
  printf("%c\n",s[i]);
} 
*/

// testing a wide-character string
wchar_t *wcs = L"123 δοκιμή";
int wcl = wcslen(wcs);

setlocale(LC_ALL, "C.UTF-8"); // (or "" if default is utf-8), needed to output wide chars
// the initial C locale blocks non-ascii chars

wprintf(L"string is [%ls], length=%d\n",wcs,wcl);
for (int i=0;i<wcl;i++) {
  wprintf(L"%lc\n",wcs[i]);
} 


/*
setlocale(LC_ALL, "C.UTF-8"); // (or "" if default is utf-8), needed to output wide chars
// the initial C locale blocks non-ascii chars

// testing a (multi)byte string
char *s = "123 δοκιμή";

int orientation_before = fwide(stdout,0);

// NOTE: cannot use printf and wprintf together!
//printf("string is [%s], length=%lu\n",s,strlen(s)); // string displayed ok, length wrong
wprintf(L"string is [%s]\n",s);	// s is converted to wchar_t *

int orientation_after = fwide(stdout,0);
wprintf(L"stdout orientation before=%d after=%d\n",orientation_before,orientation_after); 
// 0 = no orientation, 1 = wide-oriented 2 = byte-oriented 

wprintf(L"length=%lu\n",strlen(s));  // wrong length
//wprintf(L"length=%lu\n",wcslen(s)); // cannot use wcslen on char *

mbstate_t mbstate;
if (!mbsinit(&mbstate)) memset (&mbstate,0,sizeof(mbstate));
size_t wl = mbsrtowcs(NULL, (const char **)&s, 0, &mbstate); // convert and count size
wprintf(L"mbs length=%lu\n",wl);

if (!mbsinit(&mbstate)) memset (&mbstate,0,sizeof(mbstate));
wchar_t *ws = (wchar_t *)malloc(sizeof(wchar_t)*(wl+1));  // plus room for terminating 0
mbsrtowcs(ws, (const char **)&s, wl+1, &mbstate); // convert and store
wprintf(L"wc string is [%ls]\n",ws);
free(ws);

// testing a wide-character string
wchar_t *wcs = L"123 δοκιμή";
wprintf(L"string is [%ls]\n",wcs);
wprintf(L"wcs length=%lu\n",wcslen(wcs));

*/
  return 0;
}

