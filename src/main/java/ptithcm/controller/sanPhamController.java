package ptithcm.controller;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import javax.transaction.Transactional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import ptithcm.entity.DanhGiaEntity;
import ptithcm.entity.GioHangEntity;
import ptithcm.entity.NguoiDungEntity;
import ptithcm.entity.SanPhamEntity;
import ptithcm.entity.YeuThichEntity;
import ptithcm.service.DanhGiaService;
import ptithcm.service.SanPhamService;
import ptithcm.service.gioHangService;
import ptithcm.service.yeuThichService;
@Transactional
@Controller
@RequestMapping()
public class sanPhamController {
	
	@Autowired
	SanPhamService sanPhamService;
	@Autowired
	DanhGiaService danhGiaService;
	@Autowired
	gioHangService gioHangService;
	@Autowired
	yeuThichService yeuThichService;
	@RequestMapping("/product/{maSp}")
	public String sanPham(@PathVariable("maSp") String maSp, ModelMap model,HttpServletRequest request) {
		SanPhamEntity sanPham=sanPhamService.laySanPham(maSp);
		
		List<String> sizes = sanPhamService.laySizeTheoTenSanPham(maSp);
		Collections.sort(sizes, new SizeComparator());
		model.addAttribute("sizes", sizes);
		
		List<SanPhamEntity> sanPhamCungKieu = sanPhamService.laySanPhamCungKieu(maSp);
		sanPhamCungKieu = sanPhamService.locSanPhamTrung(sanPhamCungKieu);
		model.addAttribute("sanPhamCungKieu", sanPhamCungKieu);
		List<DanhGiaEntity> listDanhGia = danhGiaService.layDanhGiaSanPham(maSp);
		int count = listDanhGia.size();
		List<Map<String, Integer>> dsSoLuongVoiSize = sanPhamService.laySizeVaSoLuongTheoMaSP(maSp);
		model.addAttribute("dsSoLuongVoiSize", dsSoLuongVoiSize);

		model.addAttribute("sanPham", sanPham);
		model.addAttribute("count",count);
		model.addAttribute("danhGiaList",listDanhGia);

		return "/sanPham/sanPham";
	}



    @RequestMapping(value = "/laySoLuongTheoSize", method = RequestMethod.GET)
    @ResponseBody
    public Map<String, Integer> laySoLuongTheoSize(
            @RequestParam("size") String size,
            @RequestParam("maSP") String maSp) {
        Map<String, Integer> response = new HashMap<>();
        Integer soLuong = sanPhamService.laySoLuongTheoSize(maSp, size);
        response.put("quantity", soLuong != null ? soLuong : 0); // Default to 0 if null
        return response;
    }
//    @RequestMapping(value = "/laySoLuongTheoSize")
//    public Map<String, Integer> laySoLuongTheoSize(
//            @RequestParam("size") String size,
//            @RequestParam("maSP") String maSp) {
//        Map<String, Integer> response = new HashMap<>();
//        Integer soLuong = sanPhamService.laySoLuongTheoSize(maSp, size);
//        response.put("quantity", soLuong != null ? soLuong : 0); // Default to 0 if null
//        return response;
//    }
    
	@RequestMapping(value="themVaoGio/{maSp}")
	public String addToCart(@PathVariable("maSp") String maSp, ModelMap model, HttpServletRequest request) {
	    SanPhamEntity sanPham = sanPhamService.laySanPham(maSp);
	    HttpSession session0= request.getSession();
	    int amount = Integer.parseInt(request.getParameter("soLuong"));
	    String size = request.getParameter("size"); // Retrieve the selected size from the request
	    NguoiDungEntity user =  (NguoiDungEntity) session0.getAttribute("USER");
	    if(user == null) {
	        model.addAttribute("user", new NguoiDungEntity());
	        System.out.println("Nguoi dung moi");
	        return "user/login";
	    }
	    List<GioHangEntity> productsListInCart = gioHangService.layGioHangCuaUser(user.getMaNd());
	    boolean alreadyInCart = false;
	 // Kiểm tra xem sản phẩm đã có trong giỏ hàng chưa
	    for(int i = 0; i < productsListInCart.size(); i++) {
	        GioHangEntity gioHang = productsListInCart.get(i);
	        if(gioHang.getSanPham().equals(sanPham)) {
	        	// Nếu sản phẩm và kích cỡ đã có trong giỏ hàng, cập nhật số lượng
	            if (gioHang.getSize().equals(size)) {
	                // If the size matches, update the quantity
	                gioHang.setSoLuong(gioHang.getSoLuong() + amount);
	                gioHangService.updateGioHang(gioHang);
	                alreadyInCart = true;
	                break;
	            } else {
	            	// Nếu sản phẩm đã có nhưng kích cỡ khác, thêm mục mới vào giỏ hàng
	                GioHangEntity newGioHang = new GioHangEntity();
	                newGioHang.setNguoiDung(user);
	                newGioHang.setSanPham(sanPham);
	                newGioHang.setSoLuong(amount);
	                newGioHang.setSize(size);
	                gioHangService.addGioHang(newGioHang);
	                alreadyInCart = true;
	                break;
	            }
	        }
	    }
	    if(!alreadyInCart) {
	    	// Nếu sản phẩm chưa có trong giỏ hàng, thêm sản phẩm mới vào giỏ
	        GioHangEntity gioHang = new GioHangEntity();
	        gioHang.setNguoiDung(user);
	        gioHang.setSanPham(sanPham);
	        gioHang.setSoLuong(amount);
	        gioHang.setSize(size);
	        gioHangService.addGioHang(gioHang);
	    }
	    System.out.println("Dang them vao gio");
	    model.addAttribute("sanPham", sanPham);
	    return "sanPham/sanPham";
	}


	@RequestMapping("themVaoYT/{maSP}")
	public String addYeuThich(@PathVariable("maSP") String maSp, ModelMap model, HttpServletRequest request) {
		SanPhamEntity sanPham = sanPhamService.laySanPham(maSp);
		HttpSession session0= request.getSession();
		NguoiDungEntity user =  (NguoiDungEntity) session0.getAttribute("USER");
		if(user == null) {
			model.addAttribute("user", new NguoiDungEntity());
			System.out.println("Nguoi dung moi");
			return "user/login"; 
		}
		System.out.println("them vao YT");
		List<YeuThichEntity> yeuThichList = yeuThichService.layDSYeuThichCuaUser(user.getMaNd());
		boolean already = false;
		for(int i = 0; i < yeuThichList.size(); i++) {
			if(yeuThichList.get(i).getSanPham() == sanPham) {
				already = true;
				break;
			}
		}
		if(already ==false) {
			YeuThichEntity yeuThich = new YeuThichEntity();
			yeuThich.setNguoiDung(user);
			yeuThich.setSanPham(sanPham);
			yeuThichService.addYeuThich(yeuThich);
		}
		List<String> sizes = sanPhamService.laySizeTheoTenSanPham(maSp);
		model.addAttribute("sizes", sizes);
		
		List<SanPhamEntity> sanPhamCungKieu = sanPhamService.laySanPhamCungKieu(maSp);
		model.addAttribute("sanPhamCungKieu", sanPhamCungKieu);
		model.addAttribute("sanPham",sanPham);
		model.addAttribute(yeuThichList);
		return "sanPham/sanPham";
	}
	
	public class SizeComparator implements Comparator<String> {
	    @Override
	    public int compare(String size1, String size2) {
	        // Xác định thứ tự ưu tiên của các size
	        List<String> sizeOrder = List.of("S", "M", "L", "XL", "XXL");

	        int index1 = sizeOrder.indexOf(size1);
	        int index2 = sizeOrder.indexOf(size2);

	        // So sánh dựa trên thứ tự ưu tiên
	        return Integer.compare(index1, index2);
	    }
	}
}
