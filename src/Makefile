HDRS := $(wildcard *.h)
SRCS := $(wildcard *.cc)
OBJS := $(addprefix ../obj/, $(notdir $(SRCS:.cc=.o)))

CC := g++
CFLAGS := -march=native -mtune=native -O3 -std=c++17 -fopenmp -pedantic -Wall -DARMA_DONT_USE_WRAPPER
LDFLAGS := -lboost_program_options -llapack -lblas -lfmt -fopenmp

../main: $(OBJS) $(HDRS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

../obj/%.o: %.cc $(HDRS)
	$(CC) $(CFLAGS) -I./ -c $< -o $@

clean:
	rm $(OBJS)
