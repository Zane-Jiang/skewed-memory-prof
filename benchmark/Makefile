CFLAGS =  -Wall -L/${PCXL_ROOT}/lib -lInstruRuntime -g

TARGETS = array pointer_chase skewed_access

all: $(TARGETS)

array: array.c
	$(CC) $(CFLAGS) -o $@ $< 

pointer_chase: pointer_chase.c
	$(CC) $(CFLAGS) -o $@ $<

skewed_access: skewed_access.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) 