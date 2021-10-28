import static org.junit.Assert.assertTrue;
import org.junit.Test;
import java.util.Map;
import java.util.HashMap;
import custom.CustomModel;

public class CustomModelReturnTypeTest {

 @Test
 public void checkCustomModelReturn() {
   CustomModel cmBinary = new CustomModel("binary");
   CustomModel cmString = new CustomModel("string");
   String mimetype = "text/plain";
   String charset = "UTF-8";
   Map<String, String> queryBinary = new HashMap();
   queryBinary.put("ret_mode", "binary");
   Map<String, String> queryString = new HashMap();
   byte[] r = {116, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116};
   assertTrue(cmBinary.predict_unstructured(r, mimetype, charset, queryBinary) instanceof byte[]);
   assertTrue(cmString.predict_unstructured(r, mimetype, charset, queryString) instanceof String);
}


}