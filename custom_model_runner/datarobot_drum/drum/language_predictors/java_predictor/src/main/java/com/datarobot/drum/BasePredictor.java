package com.datarobot.drum;

import java.io.IOException;

public abstract class BasePredictor {
    protected String name;

    public BasePredictor(String name) {
        this.name = name;
    }

    public BasePredictor setName(String name) {
        this.name = name;
        return this;
    }
}
