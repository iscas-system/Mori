.PHONY: json all

default: all

CC = clang++
STD = c++17

json:
	@$(CC) -I build -I . -std=$(STD) -shared -fPIC -o build/libmori_exporter_events_json.so exporters/json_exporter/events_exporter.cpp
	@$(CC) -I build -I . -std=$(STD) -shared -fPIC -o build/libmori_exporter_tensors_json.so exporters/json_exporter/tensors_exporter.cpp
	@$(CC) -I build -I . -std=$(STD) -shared -fPIC -o build/libmori_exporter_schedule_json.so exporters/json_exporter/schedule_exporter.cpp

all: json