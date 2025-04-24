package ptithcm.controller;

import java.util.Collections;
import java.util.Comparator;
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
public class quanLiNhapHangController {
	@Autowired
	SanPhamService sanPhamService;
	
	@RequestMapping("/nhapHang")
	public String sanPham(ModelMap model,HttpServletRequest request) {


		return "/sanPham/nhapHang";
	}
}