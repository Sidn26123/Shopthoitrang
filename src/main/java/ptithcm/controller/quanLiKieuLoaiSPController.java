package ptithcm.controller;

import java.io.IOException;
import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import javax.transaction.Transactional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import ptithcm.entity.KieuSanPhamEntity;
import ptithcm.entity.LoaiSanPhamEntity;
import ptithcm.service.SanPhamService;
import ptithcm.service.loaiSanPhamService;
import ptithcm.service.KieuSanPhamService;
@Transactional
@Controller
@RequestMapping(value = "admin/brands/")
public class quanLiKieuLoaiSPController {
	
	@Autowired
	KieuSanPhamService KieuService;
	@Autowired
	loaiSanPhamService LoaiService;
	
	@RequestMapping(value = "addKieu", method = RequestMethod.GET)
	public String viewAddKieu(@ModelAttribute("brandForm") KieuSanPhamEntity brand, ModelMap model){
		List<LoaiSanPhamEntity> listLoai = LoaiService.layLoai();
		model.addAttribute("listLoai", listLoai);
		 
		return "admin/addKieu";
	}
	
	@RequestMapping(value = "addKieu", params = "add", method = RequestMethod.POST)
	public String addKieu(@ModelAttribute("brandForm") KieuSanPhamEntity brand, ModelMap model) throws IOException {
		
		try {
	        KieuService.themKieu(brand);
	        model.addAttribute("successMessage", "Thêm thành công.");
	    } catch (Exception e) {
	        model.addAttribute("errorMessage", "Có lỗi xảy ra." + e.getMessage());
	        return "admin/addKieu";
	    }
		
		return "admin/addKieu";
	}
	
	
	@RequestMapping(value = "editKieu/{maKieu}", method = RequestMethod.GET)
	public String viewEditKieu(@PathVariable("maKieu") int maKieu, @ModelAttribute("brandForm") KieuSanPhamEntity brand, ModelMap model, HttpServletRequest request) {
		 KieuSanPhamEntity Kieu = KieuService.layKieuTheoMa(maKieu);
		 model.addAttribute("Kieu", Kieu);
		 
		 List<LoaiSanPhamEntity> listLoai = LoaiService.layLoai();
		 model.addAttribute("listLoai", listLoai);
		    
		 return "admin/editKieu";
	}
	

	@RequestMapping(value = "editKieu/{maKieu}", params = "update", method = RequestMethod.POST)
	public String editKieu(@PathVariable("maKieu") int maKieu, @ModelAttribute("brandForm") KieuSanPhamEntity brand, ModelMap model, HttpServletRequest request) throws IOException {
		try {
	        KieuService.updateKieu(brand);
	        model.addAttribute("successMessage", "Cập nhật thành công.");
	    } catch (Exception e) {
	        model.addAttribute("errorMessage", "Có lỗi xảy ra khi cập nhật." + e.getMessage());
	        return "admin/editKieu";
	    }
		
		return "admin/editKieu";
	}
	
	@RequestMapping(value = "addLoai", method = RequestMethod.GET)
	public String viewAddLoai(@ModelAttribute("brandForm") LoaiSanPhamEntity brand, ModelMap model){
		return "admin/addLoai";
	}
	
	@RequestMapping(value = "addLoai", params = "add", method = RequestMethod.POST)
	public String addLoai(@ModelAttribute("brandForm") LoaiSanPhamEntity brand, ModelMap model) throws IOException {
		
		try {
	        LoaiService.themLoai(brand);
	        model.addAttribute("successMessage", "Thêm thành công.");
	    } catch (Exception e) {
	        model.addAttribute("errorMessage", "Có lỗi xảy ra." + e.getMessage());
	        return "admin/addLoai";
	    }
		
		return "admin/addLoai";
	}
	
	
	@RequestMapping(value = "editLoai/{maLoai}", method = RequestMethod.GET)
	public String viewEditLoai(@PathVariable("maLoai") String maLoai, ModelMap model, HttpServletRequest request) {
		 LoaiSanPhamEntity Loai = LoaiService.layLoaiTheoMa(maLoai);
		 model.addAttribute("Loai", Loai);
		 return "admin/editLoai";
	}
	

	@RequestMapping(value = "editLoai/{maLoai}", params = "update", method = RequestMethod.POST)
	public String editLoai(@PathVariable("maLoai") String maLoai, ModelMap model, HttpServletRequest request) throws IOException {
		
		String maLoai1=request.getParameter("maLoai");
		String tenLoai=request.getParameter("tenLoai");
		LoaiSanPhamEntity Loai = LoaiService.layLoaiTheoMa(maLoai1);
		model.addAttribute("Loai", Loai);
		Loai.setMaLoai(maLoai1);
		Loai.setTenLoai(tenLoai);
		
		try {
	        LoaiService.updateLoai(Loai);
	        model.addAttribute("successMessage", "Cập nhật thành công.");
	    } catch (Exception e) {
	        model.addAttribute("errorMessage", "Có lỗi xảy ra khi cập nhật." + e.getMessage());
	        return "admin/editLoai";
	    }
		
		return "admin/editLoai";
	}
	
	
}
