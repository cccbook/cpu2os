void _write(int fd, char *message, int len);

int _strlen(char *str) {
	char *p = str;
	while (*p) p++;
	return p-str;
}

void _puts(char *str) {
	int len = _strlen(str);
	_write(1, str, len);
}

int main() {
	_puts("Hello World!\n");
}
