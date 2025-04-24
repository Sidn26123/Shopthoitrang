package ptithcm.controller;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import javax.transaction.Transactional;

import org.hibernate.Hibernate;
import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import ptithcm.entity.NguoiDungEntity;
import ptithcm.entity.SanPhamEntity;
import ptithcm.entity.CTDonHangEntity;
import ptithcm.entity.ThongBaoEntity;
import ptithcm.entity.LoaiSanPhamEntity;
import ptithcm.service.DonHangService;
import ptithcm.service.SanPhamService;
import ptithcm.service.loaiSanPhamService;
import ptithcm.service.thongBaoService;

import ptithcm.common.timeHelper;

@Transactional
@Controller
public class mainController {

	@Autowired
	loaiSanPhamService loaiSanPhamService;

	@Autowired
	SanPhamService sanPhamService;
	@Autowired
	DonHangService DonHangService;
	@Autowired
	thongBaoService tbService;
	@RequestMapping()
	public String main(HttpServletRequest request, ModelMap model) throws IOException {

		List<LoaiSanPhamEntity> loaiSPNam = loaiSanPhamService.layLoaiTheoGioiTinh("nam");
		model.addAttribute("loaiSPNam", loaiSPNam);
		List<LoaiSanPhamEntity> loaiSPNu = loaiSanPhamService.layLoaiTheoGioiTinh("nữ");
		model.addAttribute("loaiSPNu", loaiSPNu);
	
		List<SanPhamEntity> listSpNam = sanPhamService.laySanPhamTheogioiTinh("nam");
		listSpNam = sanPhamService.locSanPhamTrung(listSpNam);
		model.addAttribute("listSpNam", listSpNam);
		List<SanPhamEntity> listSpNu = sanPhamService.laySanPhamTheogioiTinh("nữ");
		listSpNu = sanPhamService.locSanPhamTrung(listSpNu);
		model.addAttribute("listSpNu", listSpNu);
		model.addAttribute("user", new NguoiDungEntity());
		
		List<SanPhamEntity> listMoi = sanPhamService.laySanPhamMoi();
		listMoi = sanPhamService.locSanPhamTrung(listMoi);
		model.addAttribute("sanPhamMoi", listMoi);

		List<CTDonHangEntity> listPhoBien = sanPhamService.laySanPhamPhoBien(12);
		for(int i=0; i<listPhoBien.size();i++)
		{
			System.out.print(listPhoBien.get(i).getMaSP());
		}
		String absolutePath = mainController.class.getResource("").getPath();
		System.out.println("abs: " + absolutePath);
		NguoiDungEntity nd = (NguoiDungEntity) request.getSession().getAttribute("USER");
		try {
			if (nd != null && nd.getMaNd() > 0) {
				List<ThongBaoEntity> listThongBao = tbService.LayThongBaoCuaUser(nd.getMaNd());
				model.addAttribute("notifications", listThongBao);

				model.addAttribute("notificationsCount", tbService.getNumNotificationUnread(listThongBao));
			}
		}
		catch (Exception ex) {
			System.out.println(ex);
		}
		listPhoBien = sapXepVaTinhTong(listPhoBien); 
		model.addAttribute("listPhoBien", listPhoBien);
		
		return "main";
	}
	
	public List<CTDonHangEntity> sapXepVaTinhTong(List<CTDonHangEntity> sanPhamList) {
		Map<String, Integer> tongSoLuongMap = new HashMap<>();

        // Lọc và tính tổng số lượng cho từng sản phẩm cùng maSP
        for (CTDonHangEntity ctdh : sanPhamList) {
            SanPhamEntity sanPham = ctdh.getSanPham();
            int soLuong = ctdh.getSoLuong();
            
            // Kiểm tra xem sản phẩm đã tồn tại trong Map chưa
            if (tongSoLuongMap.containsKey(sanPham.getMaSP())) {
                int tongSoLuongHienTai = tongSoLuongMap.get(sanPham.getMaSP());
                tongSoLuongMap.put(sanPham.getMaSP(), tongSoLuongHienTai + soLuong);
            } else {
                tongSoLuongMap.put(sanPham.getMaSP(), soLuong);
            }
        }

        // Tạo danh sách mới để chứa các đối tượng đã lọc
        List<CTDonHangEntity> sanPhamDaLoc = new ArrayList<>();

        // Duyệt qua danh sách gốc và thêm các đối tượng đã lọc vào danh sách mới
        for (CTDonHangEntity ctdh : sanPhamList) {
            SanPhamEntity sanPham = ctdh.getSanPham();
            boolean checkSPTrung = false;
            for(int i  = 0 ; i < sanPhamDaLoc.size(); i++) {
            	if (sanPhamDaLoc.get(i).getMaSP() == sanPham.getMaSP()) {
            		checkSPTrung = true;
            	}
            }
            if(checkSPTrung ) {
            	continue;
            }
            int soLuong = tongSoLuongMap.get(sanPham.getMaSP());

            // Tạo một đối tượng mới với tổng số lượng
            CTDonHangEntity ctdhMoi = new CTDonHangEntity();
            ctdhMoi.setSanPham(sanPham);
            ctdhMoi.setSoLuong(soLuong);

            sanPhamDaLoc.add(ctdhMoi);
        }
        
        // Sử dụng Collections.sort để sắp xếp danh sách theo số lượng giảm dần
        Collections.sort(sanPhamDaLoc, (ctdh2, ctdh1) -> Integer.compare(ctdh1.getSoLuong(), ctdh2.getSoLuong()));
	    return sanPhamDaLoc;
	}
	
	@RequestMapping("khongCoQuyen")
	public String khongCoQuyen() {
		return "redirect:/user/login.htm";
	}

}
