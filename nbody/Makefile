
all: nbody nbodympi

nbody:	nbody.c
	cc -o $@ $< -lm

nbodympi:	nbodympi.c
	mpicc -o $@ $< -lm

clean:
	rm -f nbody nbodympi
