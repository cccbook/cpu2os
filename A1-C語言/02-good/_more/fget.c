int fgetc(FILE *stream) {
	char c;
	int n = fread(&c, 1, 1, stream);
	if (n <= 0) return EOF;
	return (int) c;
}
