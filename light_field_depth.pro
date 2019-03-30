TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG += qt
CONFIG -= gui

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

QMAKE_CXXFLAGS += -fopenmp -std=c++17

SOURCES += \
        main.cpp \
    lf_container.cpp \
    maxflow/graph.cpp \
    maxflow/maxflow.cpp \
    ledepthestimator.cpp \
    pmdepthestimator.cpp

HEADERS += \
        mainwindow.h \
    lf_container.h \
    Plane.h \
    gnuplot-iostream.h \
    ledepthestimator.h \
    pmdepthestimator.h

LIBS += -lhdf5_cpp -lhdf5 -fopenmp
LIBS += -lboost_iostreams -lboost_system -lboost_filesystem
