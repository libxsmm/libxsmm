#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if 0
#ifndef USE_LIBXSMM_HEADERS
  #include "header_names.h"
#else
  #include "generator_common.h"
#endif
#endif

#define SEPCHARS " .,?\"\n"

void convert_int_to_name ( char *line, unsigned int *ret )
{
   char ctmp, *word, command[80];
   FILE *fp;
   int i, size;

   sprintf(command,"grep %u ../../src/generator_common.h | cut -b 27- > /tmp/gmh1\n",*ret);
   system(command);
   fp = fopen("/tmp/gmh1","r");
   if ( fp == NULL )
   {
      fprintf(stderr,"fopen in convert_int_to_name() failed to open a file!\n");
      exit(-1);
   }
   fseek ( fp, 0, SEEK_END );
   size = ftell(fp);
   if ( size == 0 ) {
      fclose(fp);
      sprintf(command,"grep 0x%08x ../../src/generator_common.h | cut -b 27- > /tmp/gmh1\n",*ret);
      system(command);
      fp = fopen("/tmp/gmh1","r");
      if ( fp == NULL ) { fprintf(stderr,"fopen problem\n"); exit(-1); }
      fseek(fp, 0, SEEK_END );
      size = ftell(fp);
      if ( size == 0 ) {
         fprintf(stderr,"unable to find %d=0x%08x inside generator_common.h\n",*ret,*ret);
         exit(-1);
      }
   }
   fseek ( fp, 0, SEEK_SET );
   if ( fgets( command, 80, fp ) == NULL )
   {
      fprintf(stderr,"convert_int_to_name couldn't read any lines\n");
   }
   fclose(fp);
   word = strtok ( command, SEPCHARS );
   i = 0;
   while ( i < strlen(word) ) {
      ctmp = word[i];
      if ( (ctmp >= 'A') && (ctmp <= 'Z') ) ctmp = ctmp - 'A' + 'a';
      line[i] = ctmp;
      ++i;
   }
   line[i] = '\0';
   return ;
}
