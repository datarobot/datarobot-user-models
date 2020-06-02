package com.datarobot.custom;

import java.io.IOException;

abstract class BasePredictor {
    protected String name;

    public BasePredictor(String name) {
        this.name = name;
    }

    public BasePredictor setName(String name) {
        this.name = name;
        return this;
    }
}
