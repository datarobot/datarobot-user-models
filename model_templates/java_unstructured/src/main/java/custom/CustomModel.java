package custom;

import com.datarobot.drum.*;
import java.util.HashMap;
import java.util.Map;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

import java.io.ObjectOutputStream;
import java.io.DataOutputStream;
import java.io.ByteArrayOutputStream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class CustomModel extends BasePredictor { 

    String customModelPath = null;
    private ObjectMapper objectMapper = new ObjectMapper();

    public CustomModel(String name) {
        super(name);
    }

    @Override
    public void configure(Map<String, Object> params) throws Exception {
        customModelPath = (String) params.get("__custom_model_path__");
    }

    @Override
    public String predict(byte[] inputBytes) throws Exception {
        throw new Exception("NOT IMPLEMENTED");
    }

    @Override
    public <T> T predictUnstructured(byte[] inputBytes, String mimetype, String charset, Map<String, String> query) {
        System.out.println("incoming mimetype: " + mimetype);
        System.out.println("Incoming Charset: " + charset);
        try {
            System.out.println("Incoming Query: " + objectMapper.writeValueAsString(query));
        } catch (JsonProcessingException e) { 
            e.printStackTrace();
        }
        
        String retMode = query.getOrDefault("ret_mode", "text");
        String s = null;
        try { 
            s = new String(inputBytes, charset);
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        System.out.println("Incoming data: " + s );

        Integer count = s.replace("\n", " ").replace("  ", " ").split(" ").length;
        System.out.println(count.intValue());

        switch (retMode) {
            case "binary":
                byte[] retBytes = null;
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                // ObjectOutputStream out = null;
                DataOutputStream dos = null;
                try {
                    dos = new DataOutputStream(bos);                 
                    dos.writeInt( count );
                    dos.flush();
                    retBytes = bos.toByteArray();
                } catch (IOException e) { 
                    e.printStackTrace() ;
                } finally {
                    try {
                        bos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                return (T) retBytes;
            default:
                String retString = null;
                try {
                    retString = objectMapper.writeValueAsString(count);;
                } catch (JsonProcessingException e) {
                    e.printStackTrace();
                }
                return (T) retString;  
        }
    }
}

