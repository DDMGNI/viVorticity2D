
PYTHONPATH := $(CURDIR):${PYTHONPATH}
export PYTHONPATH


all:
	$(MAKE) -C vorticity


clean:
	$(MAKE) clean -C vorticity

